"""Intent Router 的架构无关输入输出模型。"""

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

Intent = Literal["chat", "create"]
RouteSource = Literal["explicit", "embedding", "llm", "fallback"]
RequestedAction = Literal["create"]


@dataclass(slots=True)
class RouteContext:
    user_message: str
    requested_action: RequestedAction | None = None
    active_ppt_filename: str | None = None
    style: str = "business"
    recent_messages: list[BaseMessage] = field(default_factory=list)


class RouteDecision(BaseModel):
    intent: Intent
    source: RouteSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str


class LLMIntentDecision(BaseModel):
    intent: Intent = Field(description="用户意图，只能是 chat 或 create")
    reason: str = Field(description="一句话分类理由")
