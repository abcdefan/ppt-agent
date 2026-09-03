"""统一智能助手请求模型。"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

AssistantAction = Literal["create"]
PptStyle = Literal["business", "creative", "academic", "minimalist"]


class AssistantStreamRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str | None = Field(default=None, description="会话 ID")
    requested_action: AssistantAction | None = Field(
        default=None,
        description="前端创建模式显式指定的 create 意图",
    )
    style: PptStyle | None = Field(default=None, description="可选 PPT 风格")


class ResumeRequest(BaseModel):
    """Create 断点重跑或 Edit HITL 选择恢复请求。"""

    run_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="需要恢复的 Workflow Run ID",
    )
    ppt_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Edit HITL 中用户选择的 PPT ID",
    )
    revision: int | None = Field(
        default=None,
        ge=0,
        description="Edit Run 进入等待状态时的乐观锁版本",
    )

    @model_validator(mode="after")
    def validate_edit_selection(self):
        """Edit 选择必须同时提供 ppt_id 和 revision。"""
        if (self.ppt_id is None) != (self.revision is None):
            raise ValueError("ppt_id 和 revision 必须同时提供")
        return self


# 保留旧名称，避免已有 Python 调用方在过渡期内失效。
CreateResumeRequest = ResumeRequest
