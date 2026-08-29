"""当前 Edit Workflow 的规划与路由节点。

Create 已改为 Enhancement Planner 加确定性子图，不依赖本模块。
Edit 的完整 Planner/Dispatcher 设计将在后续单独演进；这里暂时只保留
当前已经实现的 Assets 与 Beautify 能力。
"""

from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.agents.workflow.state import AssetTask, WorkflowState

EditStage = Literal["assets", "beautify"]
EditRoute = Literal["assets", "beautify", "FINISH"]

EDIT_STAGE_ORDER: tuple[EditStage, ...] = ("assets", "beautify")
EDIT_SUPERVISOR_NODE = "edit_supervisor_node"

EDIT_SUPERVISOR_PROMPT = """你是 PPT 多智能体团队当前版本的 Edit Supervisor。
你只负责规划和路由已有 PPT 的图片、图表与视觉美化修改，不直接修改 PPT 文件。

可选择的阶段：
- assets：准备图片或图表操作，再由 PPT Writer 统一写入 PPT；
- beautify：对已有 PPT 做视觉优化；
- FINISH：本轮所有必需阶段已经完成。

首次规划时必须返回 required_stages：
- 修改或替换图片、图表时选择 assets；
- 调整视觉样式和排版时选择 beautify；
- 当前版本不要返回其他阶段。

如果 required_stages 包含 assets，首次规划还必须返回 asset_tasks：
- 图片或插图任务加入 image；
- 图表或数据可视化任务加入 chart；
- 两类都需要时同时加入 image 和 chart；
- 不需要 assets 时返回空列表。

后续路由规则：
1. required_stages 初始化后不可修改；
2. 不要重复已完成阶段；
3. 同时需要 assets 和 beautify 时，先执行 assets；
4. 全部必需阶段完成后才能选择 FINISH。

严格返回 JSON，字段为 next、reason、required_stages、asset_tasks。
首次规划示例：
{"next":"assets","reason":"用户要求替换配图","required_stages":["assets"],"asset_tasks":["image"]}
后续路由示例：
{"next":"FINISH","reason":"图片修改已经完成","required_stages":null,"asset_tasks":null}
"""


class EditRouteDecision(BaseModel):
    """Edit Supervisor LLM 提议的路由结果。"""

    next: EditRoute = Field(description="下一阶段，或 FINISH")
    reason: str = Field(
        default="Edit Supervisor 未提供路由理由",
        description="选择该路由的一句话理由",
    )
    required_stages: list[EditStage] | None = Field(
        default=None,
        description="仅首次规划时返回；后续保持为 null",
    )
    asset_tasks: list[AssetTask] | None = Field(
        default=None,
        description="仅首次规划 Assets 时返回 image、chart 或两者",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_common_route_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "next" not in normalized and "next_stage" in normalized:
            normalized["next"] = normalized.pop("next_stage")
        if not normalized.get("reason"):
            normalized["reason"] = "Edit Supervisor 未提供路由理由"
        return normalized


def normalize_edit_stages(proposed: list[EditStage] | None) -> list[EditStage]:
    """去重并固定当前 Edit 阶段顺序。"""
    selected = set(proposed or [])
    return [stage for stage in EDIT_STAGE_ORDER if stage in selected]


def normalize_asset_tasks(
    proposed: list[AssetTask] | None,
    *,
    assets_required: bool,
) -> list[AssetTask]:
    if not assets_required:
        return []
    selected = set(proposed or ["image", "chart"])
    return [task for task in ("image", "chart") if task in selected]


def _edit_stage_completed(state: WorkflowState, stage: EditStage) -> bool:
    if stage not in state.get("completed_stages", []):
        return False
    if stage == "assets":
        return state.get("asset_apply_status") in {"succeeded", "skipped"}
    return True


def _next_safe_edit_stage(
    state: WorkflowState,
    required_stages: list[EditStage],
) -> EditRoute:
    missing = [
        stage for stage in required_stages if not _edit_stage_completed(state, stage)
    ]
    if not missing:
        return "FINISH"
    if "assets" in missing:
        return "assets"
    return "beautify"


def guard_edit_route(
    state: WorkflowState,
    proposed_next: EditRoute,
    required_stages: list[EditStage],
) -> tuple[EditRoute, str | None]:
    """阻止 Edit 提前结束、重复执行或跳过 Assets 依赖。"""
    safe_next = _next_safe_edit_stage(state, required_stages)
    if proposed_next == "FINISH":
        if safe_next == "FINISH":
            return "FINISH", None
        return safe_next, "必需阶段尚未全部完成，已阻止 FINISH"

    proposed_stage = cast(EditStage, proposed_next)
    if proposed_stage not in required_stages:
        return safe_next, f"{proposed_stage} 不属于本轮 required_stages"
    if _edit_stage_completed(state, proposed_stage):
        return safe_next, f"{proposed_stage} 已完成，已阻止重复执行"
    if proposed_stage == "beautify" and "assets" in required_stages:
        if not _edit_stage_completed(state, "assets"):
            return safe_next, "assets 尚未完成，已阻止提前美化"
    return proposed_stage, None


def route_after_edit_supervisor(state: WorkflowState) -> EditRoute:
    next_route = state.get("next")
    return next_route if next_route in {"assets", "beautify", "FINISH"} else "FINISH"


def _safe_recent_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    safe_messages = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]
    return safe_messages[-8:]


def build_edit_supervisor_node(llm: BaseChatModel):
    """创建仅服务当前 Edit 子图的结构化 Supervisor Node。"""
    structured_llm = llm.with_structured_output(
        EditRouteDecision,
        method="json_mode",
    )

    async def run_edit_supervisor(state: WorkflowState) -> dict:
        initialized = state.get("requirements_initialized", False)
        existing_required = normalize_edit_stages(state.get("required_stages"))
        existing_asset_tasks = normalize_asset_tasks(
            state.get("asset_tasks"),
            assets_required="assets" in existing_required,
        )
        state_summary = (
            f"用户本轮修改要求：{state['user_message']}\n"
            f"PPT ID：{state.get('ppt_id') or '尚未解析'}\n"
            f"PPT 风格：{state['style']}\n"
            f"PPT 文件：{state.get('filename') or '尚未生成'}\n"
            f"requirements_initialized：{initialized}\n"
            f"required_stages：{existing_required if initialized else '尚未规划'}\n"
            f"asset_tasks：{existing_asset_tasks if initialized else '尚未规划'}\n"
            f"completed_stages：{state.get('completed_stages', [])}\n"
            "请规划或选择下一阶段。"
        )
        decision = await structured_llm.ainvoke(
            [
                SystemMessage(content=EDIT_SUPERVISOR_PROMPT),
                *_safe_recent_messages(state.get("messages", [])),
                HumanMessage(content=state_summary),
            ]
        )

        required_stages = (
            existing_required
            if initialized
            else normalize_edit_stages(decision.required_stages)
        )
        asset_tasks = (
            existing_asset_tasks
            if initialized
            else normalize_asset_tasks(
                decision.asset_tasks,
                assets_required="assets" in required_stages,
            )
        )
        next_route, guard_reason = guard_edit_route(
            state,
            decision.next,
            required_stages,
        )
        reason = decision.reason
        if guard_reason:
            reason = f"{reason}；Route Guard：{guard_reason}，改为 {next_route}"

        return {
            "next": next_route,
            "route_reason": reason,
            "required_stages": required_stages,
            "asset_tasks": asset_tasks,
            "requirements_initialized": True,
        }

    return run_edit_supervisor
