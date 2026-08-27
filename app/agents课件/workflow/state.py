"""workflow 模式图状态定义 — Supervisor 与各 specialist 共享的 AgentState"""

from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Supervisor 多智能体共享状态。

    - messages: 贯穿全图的对话流，add_messages reducer 负责追加合并
    - next: supervisor 的路由决策
    - outline: outline 专家产出后写入，research 据此拆检索问题，content 据此填充结构
    - research: research 专家产出后写入（研究笔记 JSON），content 据此为正文提供事实依据
    - filename: content 专家产出后写入，后续 image/chart/beautify 作为工具入参
    - style: PPT 主题风格，入口注入
    - user_message: 原始用户请求，便于各节点直接引用
    - session_id: 记忆读写标识
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: Optional[Literal["outline", "research", "content", "image", "chart", "beautify", "FINISH"]]
    intent: Optional[Literal["chat", "create", "enhance", "analyze"]]
    outline: Optional[str]
    research: Optional[str]
    filename: Optional[str]
    style: str
    user_message: str
    session_id: str
