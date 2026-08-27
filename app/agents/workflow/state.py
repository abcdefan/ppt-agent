"""Workflow 中所有节点共享的状态结构。

State 是比对话消息更丰富的工作流上下文。我们把大纲、文件名、路由结果等
关键业务信息定义为固定字段，由代码负责更新和传递，降低关键信息在 LLM
自然语言转述过程中丢失或被改写的风险。

``messages`` 只是 State 中负责保存对话上下文的字段。各 Specialist Agent
完成工作后，会把业务结果写回 State，供后续节点直接读取。
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Supervisor 可以选择的下一节点；FINISH 表示结束整张图。
WorkflowRoute = Literal[
    "outline",
    "research",
    "content",
    "image",
    "chart",
    "beautify",
    "FINISH",
]

WorkflowIntent = Literal["chat", "create"]
WorkflowRouteSource = Literal["explicit", "embedding", "llm", "fallback"]
WorkflowAgentRole = Literal[
    "outline", "research", "content", "image", "chart", "edit", "beautify"
]


def merge_completed_agents(
    current: list[WorkflowAgentRole],
    updates: list[WorkflowAgentRole],
) -> list[WorkflowAgentRole]:
    """合并已完成节点并保持顺序，避免同一节点被重复记录。"""
    return list(dict.fromkeys([*current, *updates]))


def merge_asset_operations(
    current: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并并行分支准备的操作，并按 operation_id 去重。"""
    merged: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for operation in [*current, *updates]:
        operation_id = operation.get("operation_id")
        if isinstance(operation_id, str) and operation_id:
            merged[operation_id] = operation
        else:
            anonymous.append(operation)
    return [*merged.values(), *anonymous]


class WorkflowState(TypedDict):
    """一次 Workflow 调用期间，由所有 Graph Node 共同读写的 State。"""

    # 对话消息上下文。add_messages 是该字段的合并函数：节点返回新消息时，
    # LangGraph 会执行 add_messages(已有消息, 新消息)，而不是覆盖已有消息。
    messages: Annotated[list[BaseMessage], add_messages]

    # 本次请求的基础信息，由 Workflow 入口写入。
    user_message: str
    session_id: str
    style: str
    requested_action: Literal["create"] | None

    # Specialist Agents 的业务产物。
    # Outline 写入 outline，Research 读取大纲并写入 research_report，
    # Content 读取两者后写入 filename；
    # Image、Chart 和 Beautify 再读取 filename 操作同一个 PPT 文件。
    outline: str | None
    research_report: str | None
    filename: str | None

    # Image/Chart 在生产子图中只准备资源并写入结构化操作；edit_node
    # 是唯一 PPT Writer，负责汇总这些操作后一次性提交。
    asset_operations: Annotated[
        list[dict[str, Any]],
        merge_asset_operations,
    ]
    asset_apply_status: Literal["pending", "succeeded", "skipped", "failed"]
    applied_operation_ids: list[str]

    # Specialist 的内部 Tool Call 消息不进入公共 messages；Supervisor
    # 通过这个结构化字段判断哪些节点已经完成。
    completed_agents: Annotated[
        list[WorkflowAgentRole],
        merge_completed_agents,
    ]

    # Supervisor 写入下一步路由，条件边根据它选择下一个节点。
    next: WorkflowRoute | None

    # Intent Router 的本轮结果；这些字段只参与执行和观测，不持久化。
    intent: WorkflowIntent | None
    route_source: WorkflowRouteSource | None
    route_confidence: float | None
    route_reason: str | None
