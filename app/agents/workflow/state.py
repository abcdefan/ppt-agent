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

# Workflow 的业务阶段。Image/Chart/PPT Writer 被封装在 assets 阶段内部。
WorkflowStage = Literal[
    "research",
    "outline",
    "content",
    "assets",
    "beautify",
]

# 当前 Edit Supervisor 使用的下一阶段；Create 只在最终持久化前写入 FINISH。
WorkflowRoute = Literal[
    "assets",
    "beautify",
    "FINISH",
]

WorkflowIntent = Literal["chat", "create", "edit"]
WorkflowRouteSource = Literal["explicit", "embedding", "llm", "fallback"]
WorkflowAgentRole = Literal[
    "research", "outline", "content", "image", "chart", "writer", "beautify"
]
AssetTask = Literal["image", "chart"]


def merge_completed_agents(
    current: list[WorkflowAgentRole],
    updates: list[WorkflowAgentRole],
) -> list[WorkflowAgentRole]:
    """合并已完成节点并保持顺序，避免同一节点被重复记录。"""
    return list(dict.fromkeys([*current, *updates]))


def merge_completed_stages(
    current: list[WorkflowStage],
    updates: list[WorkflowStage],
) -> list[WorkflowStage]:
    """合并已完成业务阶段并保持首次完成时的顺序。"""
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


def merge_attempt_counts(
    current: dict[str, int],
    updates: dict[str, int],
) -> dict[str, int]:
    """合并各节点的已尝试次数，同名节点取最新值。"""
    merged = dict(current)
    merged.update(updates)
    return merged


class WorkflowState(TypedDict):
    """一次 Workflow 调用期间，由所有 Graph Node 共同读写的 State。"""

    # 对话消息上下文。add_messages 是该字段的合并函数：节点返回新消息时，
    # LangGraph 会执行 add_messages(已有消息, 新消息)，而不是覆盖已有消息。
    messages: Annotated[list[BaseMessage], add_messages]

    # 本次请求的基础信息，由 Workflow 入口写入。
    user_id: int
    run_id: str
    user_message: str
    session_id: str
    style: str
    requested_action: Literal["create", "edit"] | None
    # requested_ppt_id 是调用方可选的显式目标；ppt_id 是初始化/解析后
    # 本轮真正操作的 PPT。active_ppt_id 仅用于 Intent Router 提示。
    requested_ppt_id: str | None
    active_ppt_id: str | None
    ppt_id: str | None
    ppt_context_error: str | None
    workflow_error: str | None

    # Specialist Agents 的业务产物。
    # Research 根据用户需求写入 research_report，Outline 读取报告后写入
    # outline，Content 再读取两者并同时写入 filename 和 slides_manifest；
    # 后续 Planner 与增强节点读取页面清单，Image、Chart 和 Beautify 再通过
    # filename 操作同一个 PPT 文件。
    outline: str | None
    research_report: str | None
    filename: str | None
    slides_manifest: list[dict[str, Any]] | None

    # Image/Chart 在生产子图中只准备资源并写入结构化操作；ppt_writer_node
    # 是唯一 PPT Writer，负责汇总这些操作后一次性提交。
    asset_operations: Annotated[
        list[dict[str, Any]],
        merge_asset_operations,
    ]
    asset_apply_status: Literal["pending", "succeeded", "skipped", "failed"]
    applied_operation_ids: list[str]
    # Create 的 Enhancement Planner 或 Edit Supervisor 写入；Assets 子图
    # 据此动态选择 Image、Chart、两者并行或跳过。
    asset_tasks: list[AssetTask]

    # Specialist 的内部 Tool Call 消息不进入公共 messages；工作流通过
    # 这个结构化字段判断哪些节点已经完成。
    completed_agents: Annotated[
        list[WorkflowAgentRole],
        merge_completed_agents,
    ]

    # Create 初始化时先写入 Research/Outline/Content，Content 生成文件并写回
    # slides_manifest 后，由 Enhancement Planner 一次性追加可选阶段；当前
    # Edit 由专用 Supervisor 规划。规划完成后通常冻结；可选增强重试耗尽并
    # 降级交付时，可以移除对应的非关键阶段。
    required_stages: list[WorkflowStage]
    completed_stages: Annotated[
        list[WorkflowStage],
        merge_completed_stages,
    ]
    requirements_initialized: bool

    # Edit Supervisor 写入下一步路由；Create 只在收尾时写入 FINISH。
    next: WorkflowRoute | None

    # 节点级重试。attempt_error 记录最近一次可重试失败的原因（达到重试
    # 上限后由 Finalize 提升为 workflow_error）；attempt_counts 按节点名
    # 记录"已尝试次数"（每次执行都 +1，含成功那次），路由据此比较
    # agent_max_attempts 上限，决定回指自身重试还是失败收尾/降级。
    attempt_error: str | None
    attempt_counts: Annotated[dict[str, int], merge_attempt_counts]

    # Intent Router 的本轮结果；这些字段只参与执行和观测，不持久化。
    intent: WorkflowIntent | None
    route_source: WorkflowRouteSource | None
    route_confidence: float | None
    route_reason: str | None
