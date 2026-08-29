"""Chat/Create 共用的统一流式入口。"""

import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.sse import SSE_HEADERS, sse_event
from app.schemas.assistant import AssistantStreamRequest, CreateResumeRequest

router = APIRouter(prefix="/assistant", tags=["智能助手"])
logger = logging.getLogger(__name__)


@router.get("/tasks")
async def list_create_tasks(request: Request):
    """返回当前用户持有的 Create PPT 任务及断点进度。"""
    agent_runner = request.app.state.agent_runner
    ppt_context_service = getattr(agent_runner, "ppt_context_service", None)
    if ppt_context_service is None:
        return {"items": []}
    items = await ppt_context_service.list_create_tasks(
        user_id=settings.local_user_id,
    )
    return {"items": items}


@router.post("/stream")
async def assistant_stream(
    req: AssistantStreamRequest,
    request: Request,
):
    """由公共 Intent Router 决定普通对话或 PPT 创建流程。"""
    user_id = settings.local_user_id
    session_id = req.session_id or f"session-{uuid4()}"
    run_id = f"run-{uuid4().hex}"
    agent_runner = request.app.state.agent_runner

    async def event_generator():
        try:
            async for event in agent_runner.run_stream(
                user_message=req.message,
                session_id=session_id,
                requested_action=req.requested_action,
                style=req.style,
                # 请求中的 ppt_id 只是 Edit 可选的已有目标，不是 Create
                # 即将生成的新 PPT ID；进入 Workflow 后统一使用该名称。
                requested_ppt_id=req.ppt_id,
                user_id=user_id,
                run_id=run_id,
            ):
                yield sse_event(event["event"], event["data"])
        except Exception as exc:
            logger.exception("[统一助手流式接口] 未捕获异常")
            yield sse_event("ERROR", {"message": f"服务器异常: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/resume")
async def resume_create_stream(
    req: CreateResumeRequest,
    request: Request,
):
    """根据已有 Run ID 从应用层持久化断点继续执行 Create Workflow。"""
    user_id = settings.local_user_id
    agent_runner = request.app.state.agent_runner

    async def event_generator():
        resume_stream = getattr(agent_runner, "run_create_resume_stream", None)
        if resume_stream is None:
            yield sse_event(
                "ERROR",
                {"message": "当前 Agent 模式不支持 Create Workflow 断点重跑"},
            )
            return
        try:
            async for event in resume_stream(
                user_id=user_id,
                run_id=req.run_id,
            ):
                yield sse_event(event["event"], event["data"])
        except Exception as exc:
            logger.exception("[Create 断点重跑接口] 未捕获异常")
            yield sse_event("ERROR", {"message": f"服务器异常: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
