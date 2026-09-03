"""Chat/Create 共用的统一流式入口。"""

import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.sse import SSE_HEADERS, sse_event
from app.schemas.assistant import AssistantStreamRequest, ResumeRequest

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
async def resume_workflow_stream(
    req: ResumeRequest,
    request: Request,
):
    """Resume Create 断点，或向 Edit HITL checkpoint 提交 PPT 选择。"""
    user_id = settings.local_user_id
    agent_runner = request.app.state.agent_runner

    async def event_generator():
        is_edit_resume = req.ppt_id is not None
        resume_stream = getattr(
            agent_runner,
            (
                "run_edit_resume_stream"
                if is_edit_resume
                else "run_create_resume_stream"
            ),
            None,
        )
        if resume_stream is None:
            yield sse_event(
                "ERROR",
                {
                    "message": (
                        "当前 Agent 模式不支持 Edit HITL 恢复"
                        if is_edit_resume
                        else "当前 Agent 模式不支持 Create Workflow 断点重跑"
                    )
                },
            )
            return
        try:
            resume_kwargs = {"user_id": user_id, "run_id": req.run_id}
            if is_edit_resume:
                resume_kwargs.update(
                    {
                        "ppt_id": req.ppt_id,
                        "revision": req.revision,
                    }
                )
            async for event in resume_stream(**resume_kwargs):
                yield sse_event(event["event"], event["data"])
        except Exception as exc:
            logger.exception("[Workflow 恢复接口] 未捕获异常")
            yield sse_event("ERROR", {"message": f"服务器异常: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# 保留原函数名，便于现有单元测试和内部调用平滑迁移。
resume_create_stream = resume_workflow_stream
