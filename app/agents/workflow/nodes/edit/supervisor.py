"""Edit Workflow 的动态规划与路由节点。"""

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.agents.workflow.state import AssetTask, EditRoute, WorkflowState

EDIT_ROUTES: set[str] = {
    "research",
    "outline",
    "content",
    "planner",
    "assets",
    "beautify",
    "FINISH",
}
EDIT_SUPERVISOR_NODE = "edit_supervisor_node"

EDIT_SUPERVISOR_PROMPT = """你是 PPT 多智能体团队的 Edit Supervisor。
你负责根据用户的修改要求和每个阶段执行后的最新 State，每轮只选择一个下一阶段；
你不直接修改 PPT，也不需要一次性规划或冻结整条执行路径。

可选择的阶段：
- research：重新检索资料并更新研究报告；
- outline：重新规划完整页面结构；
- content：根据当前研究报告和大纲重新生成基础 PPT；
- planner：根据当前 PPT 内容判断是否需要图片、图表或美化；
- assets：准备图片/图表操作，再由 PPT Writer 单点写入；
- beautify：对当前 PPT 做视觉优化；
- FINISH：本轮修改已经完成。

调度原则：
1. 结合用户原始要求、当前已有产物和已经执行过的阶段选择下一步；
2. research 更新后，由你判断是否还需要 outline 或 content；
3. outline 更新后通常还需要 content；content 重新生成后通常应调用 planner；
4. 单纯换图或图表可以直接调用 assets，单纯调整视觉风格可以直接 beautify；
5. 选择 assets 时，通过 asset_tasks 指定 image、chart 或两者；
6. 不要无理由重复已经执行成功的阶段；确认修改已经落到 PPT 后再 FINISH；
7. 节点失败由工作流直接失败收尾，不由你进行节点级重试。

严格返回 JSON，字段为 edit_next、reason、asset_tasks。
示例：
{"edit_next":"research","reason":"需要先获取用户要求的最新资料","asset_tasks":null}
{"edit_next":"assets","reason":"只需要替换页面配图","asset_tasks":["image"]}
{"edit_next":"FINISH","reason":"用户要求的修改已经写入 PPT","asset_tasks":null}
"""


class EditRouteDecision(BaseModel):
    """Edit Supervisor 每轮提议的下一步。"""

    edit_next: EditRoute = Field(description="Edit 下一阶段，或 FINISH")
    reason: str = Field(
        default="Edit Supervisor 未提供路由理由",
        description="选择该路由的一句话理由",
    )
    asset_tasks: list[AssetTask] | None = Field(
        default=None,
        description="edit_next=assets 时选择 image、chart 或两者，否则返回 null",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_common_route_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "edit_next" not in normalized and "next_stage" in normalized:
            normalized["edit_next"] = normalized.pop("next_stage")
        if not normalized.get("reason"):
            normalized["reason"] = "Edit Supervisor 未提供路由理由"
        return normalized


def normalize_asset_tasks(proposed: list[AssetTask] | None) -> list[AssetTask]:
    """去重并固定 Assets 内 Image/Chart 的执行顺序。"""
    selected = set(proposed or [])
    return [task for task in ("image", "chart") if task in selected]


def route_after_edit_supervisor(state: WorkflowState) -> EditRoute:
    next_route = state.get("edit_next")
    return cast(EditRoute, next_route) if next_route in EDIT_ROUTES else "FINISH"


def _safe_conversation_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    safe_messages = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]
    return safe_messages[-8:]


def build_edit_supervisor_node(llm: BaseChatModel):
    """创建每轮只选择一个动作的结构化 Edit Supervisor Node。"""
    structured_llm = llm.with_structured_output(
        EditRouteDecision,
        method="json_mode",
    )

    async def run_edit_supervisor(state: WorkflowState) -> dict:
        state_summary = (
            f"用户本轮修改要求：{state['user_message']}\n"
            f"PPT ID：{state.get('ppt_id') or '尚未解析'}\n"
            f"PPT 风格：{state['style']}\n"
            f"PPT 文件：{state.get('filename') or '尚未生成'}\n"
            f"研究报告是否存在：{bool(state.get('research_report'))}\n"
            f"页面大纲是否存在：{bool(state.get('outline'))}\n"
            f"页面清单是否存在：{bool(state.get('slides_manifest'))}\n"
            f"PPT 已有产物阶段：{state.get('completed_stages', [])}\n"
            f"Planner 最近建议：{state.get('required_stages', [])}\n"
            f"Planner/上轮选择的资源任务：{state.get('asset_tasks', [])}\n"
            "请根据最新状态只选择一个下一阶段。"
        )
        decision = await structured_llm.ainvoke(
            [
                SystemMessage(content=EDIT_SUPERVISOR_PROMPT),
                *_safe_conversation_history(
                    state.get("conversation_history", [])
                ),
                HumanMessage(content=state_summary),
            ]
        )

        patch: dict[str, Any] = {
            "edit_next": decision.edit_next,
            "route_reason": decision.reason,
        }
        if decision.edit_next == "assets":
            proposed_tasks = (
                decision.asset_tasks
                if decision.asset_tasks is not None
                else state.get("asset_tasks", [])
            )
            patch["asset_tasks"] = normalize_asset_tasks(proposed_tasks)
        return patch

    return run_edit_supervisor
