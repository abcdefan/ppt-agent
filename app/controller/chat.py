"""保留的会话记忆维护接口。"""

from fastapi import APIRouter, Request

from app.schemas.chat import ClearHistoryRequest
from app.schemas.common import BaseResponse

router = APIRouter(prefix="/chat", tags=["对话"])
@router.post("/clear", response_model=BaseResponse)
async def clear_history(request: ClearHistoryRequest, http_request: Request):
    """清除指定会话的历史记录"""
    agent_runner = http_request.app.state.agent_runner
    await agent_runner.clear_history(session_id=request.session_id)
    return BaseResponse.success(message="历史记录已清除")
