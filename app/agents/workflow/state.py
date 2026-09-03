"""Workflow 中所有节点共享的状态结构。

State 把“跨轮会话快照”和“本轮业务产物”分开保存：Workflow 入口只从
Redis 加载一次 conversation_history，并单独保存当前 user_message；Graph
执行期间所有节点读取同一份快照，不再维护会不断增长的公共 messages。

各 Specialist Agent 内部仍然使用自己的 messages 完成一次 ReAct 调用，但
只把提取后的业务结果写回本 State，供后续节点直接读取。
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage

# Workflow 的业务阶段。Image/Chart/PPT Writer 被封装在 assets 阶段内部。
WorkflowStage = Literal[
    "research",
    "outline",
    "content",
    "assets",
    "beautify",
]

# Edit Supervisor 每轮可选择的下一阶段。
EditRoute = Literal[
    "research",
    "outline",
    "content",
    "planner",
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

    # -------------------- 入口初始化与会话上下文 --------------------

    # 当前登录用户 ID；用于所有 MySQL 权限校验和业务记录归属判断。
    user_id: int
    # 本次 Workflow Run ID，同时作为 LangGraph checkpoint 的 thread_id。
    run_id: str
    # 当前会话 ID；Redis conversation memory 与 Session Active PPT 均按它隔离。
    session_id: str
    # Redis 在本轮进入 Graph 前加载的跨轮会话快照，不包含当前 user_message；
    # 本轮执行和 HITL 恢复期间保持不变，节点不得向其中追加内部协作消息。
    conversation_history: list[BaseMessage]
    # 当前 HTTP 请求的用户输入；与 conversation_history 分开，避免重复拼接。
    user_message: str
    # 本轮 PPT 视觉风格；未显式指定时由入口写入 business。
    style: str
    # 前端创建模式可显式指定 create；为空时由 Intent Router 结合上下文判断。
    requested_action: Literal["create"] | None
    # 当前 Session 记录的活动 PPT，仅是 Edit 目标候选，必须先经过内容核验。
    active_ppt_id: str | None

    # -------------------- Intent 路由与 Chat 回复 --------------------

    # Intent Router 的本轮分类结果。
    intent: WorkflowIntent | None
    # 用户是否明确要求本轮立即执行 Create/Edit；不确定时必须为 False。
    execute: bool
    # Intent 的决策来源：显式操作、向量匹配、LLM 或安全回退。
    route_source: WorkflowRouteSource | None
    # 路由置信度；只有能够计算置信度的来源才有值。
    route_confidence: float | None
    # 最近一次 Intent/Planner/Supervisor/降级决策的可观测理由。
    route_reason: str | None
    # Reply Node 生成的最终用户可见文本；Runner 和流式适配层从这里读取结果。
    final_response: str | None
    # streamed 表示 Chat 已逐 Token 输出；complete 表示 Create/Edit 需整段输出。
    final_response_mode: Literal["streamed", "complete"] | None

    # -------------------- Create/Edit 共用 PPT 上下文与产物 --------------------

    # 本轮实际创建或编辑的 PPT ID；入口为空，由 Initialize/Resolve Node 写入。
    ppt_id: str | None
    # PPT 初始化、解析、权限或数据恢复失败原因；Reply Node 据此生成失败回复。
    ppt_context_error: str | None
    # Research Specialist 生成或从 ppt_record 恢复的结构化调研报告。
    research_report: str | None
    # Outline Specialist 生成或从 ppt_record 恢复的页面大纲。
    outline: str | None
    # Content 生成或从 ppt_record 恢复的真实 PPTX 文件名。
    filename: str | None
    # PPT 每页真实内容清单；Planner、Assets 和持久化节点以它为准。
    slides_manifest: list[dict[str, Any]] | None

    # -------------------- Create 编排状态 --------------------

    # 本轮必须完成的业务阶段；Create Planner 可追加 Assets/Beautify。
    required_stages: list[WorkflowStage]
    # 已完成的业务阶段，使用 reducer 去重合并并保持首次完成顺序。
    completed_stages: Annotated[
        list[WorkflowStage],
        merge_completed_stages,
    ]
    # Create Enhancement Planner 是否已经运行并冻结了可选阶段计划。
    requirements_initialized: bool
    # Create Finalize 已完成最终校验；Persist 据此执行最终落库。
    create_finalized: bool

    # -------------------- Edit 目标识别与调度状态 --------------------

    # LLM 根据用户要求与 Active PPT 真实内容判断是否可直接沿用当前目标。
    edit_target_matches_active: bool | None
    # 上述保守判断的理由；先落 checkpoint，HITL 恢复时无需重新调用匹配 LLM。
    edit_target_match_reason: str | None
    # Edit Supervisor 选择的下一阶段，Create 流程不得读写该字段。
    edit_next: EditRoute | None

    # -------------------- Specialist 与 Assets 执行状态 --------------------

    # 已完成的 Specialist/Writer 角色；内部 ReAct messages 不进入公共 State。
    completed_agents: Annotated[
        list[WorkflowAgentRole],
        merge_completed_agents,
    ]
    # Planner/Supervisor 选择的资源任务；Assets 子图据此执行 image/chart。
    asset_tasks: list[AssetTask]
    # Image/Chart 并行分支准备的结构化文件写入操作，按 operation_id 合并去重。
    asset_operations: Annotated[
        list[dict[str, Any]],
        merge_asset_operations,
    ]
    # PPT Writer 对本轮 Assets 的最终处理状态。
    asset_apply_status: Literal["pending", "succeeded", "skipped", "failed"]
    # 已成功写入的资源操作 ID，用于观测和后续幂等扩展。
    applied_operation_ids: list[str]

    # -------------------- 重试与执行错误 --------------------

    # 最近一次可重试失败原因；成功、降级或提升为 workflow_error 后清空。
    attempt_error: str | None
    # 各可重试节点的已执行次数，按节点名合并最新计数。
    attempt_counts: Annotated[dict[str, int], merge_attempt_counts]
    # Specialist、Planner、Writer 等业务阶段的不可恢复错误。
    workflow_error: str | None
