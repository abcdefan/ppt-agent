"""统一智能助手请求模型。"""

from typing import Literal

from pydantic import BaseModel, Field

AssistantAction = Literal["create"]
PptStyle = Literal["business", "creative", "academic", "minimalist"]


class AssistantStreamRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str | None = Field(default=None, description="会话 ID")
    requested_action: AssistantAction | None = Field(
        default=None,
        description="前端明确指定的单次动作；V1 仅支持 create",
    )
    style: PptStyle | None = Field(default=None, description="可选 PPT 风格")
