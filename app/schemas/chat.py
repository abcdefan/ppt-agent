"""对话请求/响应模型"""

from pydantic import BaseModel, Field


class ClearHistoryRequest(BaseModel):
    """清除历史请求"""

    session_id: str = Field(..., description="要清除历史的会话 ID")
