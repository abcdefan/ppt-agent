"""统一智能助手请求模型。"""

from typing import Literal

from pydantic import BaseModel, Field

AssistantAction = Literal["create", "edit"]
PptStyle = Literal["business", "creative", "academic", "minimalist"]


class AssistantStreamRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str | None = Field(default=None, description="会话 ID")
    requested_action: AssistantAction | None = Field(
        default=None,
        description="前端明确指定的单次动作",
    )
    ppt_id: str | None = Field(
        default=None,
        min_length=1,
        description="Edit 时可选的显式目标 PPT ID；缺省时使用当前活动 PPT",
    )
    style: PptStyle | None = Field(default=None, description="可选 PPT 风格")


class CreateResumeRequest(BaseModel):
    """Create Workflow 断点重跑请求。"""

    run_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="需要恢复的 Create Workflow Run ID",
    )
