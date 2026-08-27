"""Chat/Create 共用的统一流式入口。"""

import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.sse import SSE_HEADERS, sse_event
from app.schemas.assistant import AssistantStreamRequest

router = APIRouter(prefix="/assistant", tags=["智能助手"])
logger = logging.getLogger(__name__)


@router.post("/stream")
async def assistant_stream(req: AssistantStreamRequest, request: Request):
    """由公共 Intent Router 决定普通对话或 PPT 创建流程。"""
    session_id = req.session_id or f"session-{uuid4()}"
    agent_runner = request.app.state.agent_runner

    async def event_generator():
        try:
            async for event in agent_runner.run_stream(
                user_message=req.message,
                session_id=session_id,
                requested_action=req.requested_action,
                style=req.style,
            ):
                yield sse_event(event["event"], event["data"])
        except Exception as exc:
            logger.exception("[统一助手流式接口] 未捕获异常: %s", exc)
            yield sse_event("ERROR", {"message": f"服务器异常: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
