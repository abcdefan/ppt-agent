"""在基础 PPT 生成后，根据实际页面内容规划可选增强阶段。"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.agents.workflow.state import AssetTask, WorkflowStage, WorkflowState
from app.core.config import settings

BASE_REQUIRED_STAGES: tuple[WorkflowStage, ...] = (
    "research",
    "outline",
    "content",
)

logger = logging.getLogger(__name__)

ENHANCEMENT_PLANNER_NODE = "enhancement_planner_node"

ENHANCEMENT_PLANNER_PROMPT = """你是 PPT Create Workflow 的增强规划器。
Research、Outline 和 Content 已经完成。请结合用户原始要求、调研结果、
页面大纲以及实际生成的每页内容，一次性决定是否还需要图片、图表和最终视觉美化。

规划规则：
1. 完整 PPT 通常需要合适的配图；除非用户明确要求纯文字或不要图片，否则可加入 image。
2. 只有调研报告中存在可靠、适合比较或展示的数值数据时才加入 chart，不得编造数据。
3. 用户明确表示不要图片、图表或美化时，必须遵守相应限制。
4. beautify 表示基础 PPT 和资源写入完成后的整体视觉优化。
5. 你只规划一次，不负责执行任何阶段。

严格返回 JSON，字段为 asset_tasks、beautify、reason。例如：
{"asset_tasks":["image","chart"],"beautify":true,"reason":"大纲包含人物介绍且调研报告提供了可视化数据"}
"""


class EnhancementPlan(BaseModel):
    """Create 可选增强阶段的一次性结构化规划。"""

    asset_tasks: list[AssetTask] = Field(default_factory=list)
    beautify: bool = True
    reason: str = "Enhancement Planner 未提供规划理由"

    @model_validator(mode="before")
    @classmethod
    def normalize_common_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "asset_tasks" not in normalized and "assets" in normalized:
            normalized["asset_tasks"] = normalized.pop("assets")
        if not normalized.get("reason"):
            normalized["reason"] = "Enhancement Planner 未提供规划理由"
        return normalized


def normalize_asset_tasks(tasks: list[AssetTask] | None) -> list[AssetTask]:
    """去重并固定 Image/Chart 的执行顺序。"""
    selected = set(tasks or [])
    return [task for task in ("image", "chart") if task in selected]


def build_enhancement_planner_node(llm: BaseChatModel):
    """创建只调用一次的 Enhancement Planner Node。"""
    structured_llm = llm.with_structured_output(EnhancementPlan, method="json_mode")

    async def plan_enhancements(state: WorkflowState) -> dict:
        try:
            slides_manifest = json.dumps(
                state.get("slides_manifest") or [],
                ensure_ascii=False,
            )
            decision = await structured_llm.ainvoke(
                [
                    SystemMessage(content=ENHANCEMENT_PLANNER_PROMPT),
                    HumanMessage(
                        content=(
                            f"用户原始需求：{state['user_message']}\n"
                            f"PPT 风格：{state['style']}\n"
                            f"调研报告：\n{state.get('research_report') or '无可用调研报告'}\n"
                            f"页面大纲：\n{state.get('outline') or '无可用大纲'}\n"
                            f"实际生成的 PPT 页面内容：\n{slides_manifest}\n"
                            "请规划可选增强阶段。"
                        )
                    ),
                ]
            )
        except Exception as exc:
            logger.exception("Enhancement Planner 执行失败")
            attempts = state.get("attempt_counts", {}).get("planner", 0) + 1
            if attempts < settings.agent_max_attempts.get("planner", 3):
                # 还有尝试额度：返回可重试失败，路由回指自身。
                return {
                    "attempt_counts": {"planner": attempts},
                    "attempt_error": f"增强规划失败：{exc}",
                    "requirements_initialized": False,
                    "messages": [
                        HumanMessage(
                            content=(
                                f"增强规划失败（第 {attempts} 次尝试），即将重试：{exc}"
                            ),
                            name="enhancement_planner_node",
                        )
                    ],
                }
            # 达到上限：降级为基础版本。基础 PPT 已生成，增强只是可选项，
            # 因此不标记失败，仅冻结"无增强"计划并记录降级原因。
            return {
                "asset_tasks": [],
                "required_stages": list(BASE_REQUIRED_STAGES),
                "requirements_initialized": True,
                "route_reason": f"增强规划失败，已降级为基础版本：{exc}",
                "attempt_error": None,
                "attempt_counts": {"planner": attempts},
                "messages": [
                    HumanMessage(
                        content="增强规划失败，已跳过配图/美化，交付基础版本",
                        name="enhancement_planner_node",
                    )
                ],
            }

        asset_tasks = normalize_asset_tasks(decision.asset_tasks)
        required_stages: list[WorkflowStage] = list(BASE_REQUIRED_STAGES)
        if asset_tasks:
            required_stages.append("assets")
        if decision.beautify:
            required_stages.append("beautify")

        return {
            "asset_tasks": asset_tasks,
            "required_stages": required_stages,
            "requirements_initialized": True,
            "route_reason": decision.reason,
            "attempt_error": None,
            "attempt_counts": {
                "planner": state.get("attempt_counts", {}).get("planner", 0) + 1
            },
        }

    return plan_enhancements
