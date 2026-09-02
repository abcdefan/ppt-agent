"""Intent Router 的架构无关输入输出模型。"""

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

Intent = Literal["chat", "create", "edit"]
RouteSource = Literal["explicit", "embedding", "llm", "fallback"]
RequestedAction = Literal["create", "edit"]


@dataclass(slots=True)
class RouteContext:
    user_message: str
    requested_action: RequestedAction | None = None
    active_ppt_id: str | None = None
    style: str = "business"
    recent_messages: list[BaseMessage] = field(default_factory=list)


class RouteDecision(BaseModel):
    intent: Intent
    execute: bool = Field(description="本轮是否立即执行 Create/Edit 文件操作")
    source: RouteSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str


class LLMIntentDecision(BaseModel):
    intent: Intent = Field(description="用户意图，只能是 chat、create 或 edit")
    execute: bool = Field(description="用户是否明确要求本轮立即执行该意图")
    reason: str = Field(description="一句话说明意图和执行判断理由")
