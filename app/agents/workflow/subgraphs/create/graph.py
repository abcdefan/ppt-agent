"""首次生成 PPT 的确定性 Create 子图。"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.nodes.create.finalize import (
    FINALIZE_CREATE_NODE,
    finalize_create_node,
)
from app.agents.workflow.nodes.create.planner import (
    ENHANCEMENT_PLANNER_NODE,
    build_enhancement_planner_node,
)
from app.agents.workflow.nodes.ppt_context.persist import (
    PERSIST_PPT_RECORD_NODE,
    build_persist_node,
)
from app.agents.workflow.nodes.specialist import (
    BEAUTIFY_NODE,
    CONTENT_NODE,
    OUTLINE_NODE,
    RESEARCH_NODE,
    build_specialist_node,
)
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.subgraphs.assets import (
    ASSETS_NODE,
    build_assets_subgraph,
    build_debug_assets_subgraph,
)
from app.core.config import settings
from app.services import PptContextService

PPT_CREATION_SUBGRAPH = "ppt_creation_subgraph"
DEBUG_PPT_CREATION_SUBGRAPH = "debug_ppt_creation_subgraph"

# 阶段级增量持久化：Research/Outline/Content/Assets 完成后分别记录
# 已完成阶段和业务产物；最终状态仍由 Finalize 后的 persist 决定。
PERSIST_RESEARCH_MILESTONE_NODE = "persist_research_milestone_node"
PERSIST_OUTLINE_MILESTONE_NODE = "persist_outline_milestone_node"
PERSIST_CONTENT_MILESTONE_NODE = "persist_content_milestone_node"
PERSIST_ASSETS_MILESTONE_NODE = "persist_assets_milestone_node"

PostPlannerRoute = Literal["assets", "beautify", "retry", "finalize"]
PostAssetsRoute = Literal["beautify", "finalize"]
PostBeautifyRoute = Literal["retry", "finalize"]
CreateContinueRoute = Literal["continue", "retry", "finalize"]
CreateEntryRoute = Literal[
    "research",
    "outline",
    "content",
    "planner",
    "assets",
    "beautify",
    "finalize",
]


def route_create_entry(state: WorkflowState) -> CreateEntryRoute:
    """根据已持久化的阶段和产物选择新建或恢复时的 Create 起点。

    completed_stages 是追加去重的完成阶段集合，不是仅保存最新阶段。恢复时
    先校验各阶段是否连续完成，再从最远的可靠断点倒序选择下一跳。
    """
    completed = set(state.get("completed_stages", []))
    required = set(state.get("required_stages", []))

    research_ready = (
        "research" in completed and bool(state.get("research_report"))
    )
    outline_ready = (
        research_ready
        and "outline" in completed
        and bool(state.get("outline"))
    )
    content_ready = (
        outline_ready
        and "content" in completed
        and bool(state.get("filename"))
        and bool(state.get("slides_manifest"))
    )
    assets_ready = content_ready and "assets" in completed
    beautify_ready = content_ready and "beautify" in completed
    plan_ready = content_ready and state.get("requirements_initialized", False)

    if beautify_ready:
        return "finalize"
    elif assets_ready:
        return "beautify" if "beautify" in required else "finalize"
    elif plan_ready and "assets" in required and state.get("asset_tasks"):
        return "assets"
    elif plan_ready and "beautify" in required and "assets" not in required:
        return "beautify"
    elif plan_ready and not ({"assets", "beautify"} & required):
        return "finalize"
    elif content_ready:
        # 当前没有 Planner 里程碑，恢复到 Content 后默认重新规划。
        return "planner"
    elif outline_ready:
        return "content"
    elif research_ready:
        return "outline"
    else:
        return "research"


def route_after_research(state: WorkflowState) -> CreateContinueRoute:
    """Research 产物非法时限次重试；耗尽后节点会写入降级报告。"""
    if state.get("workflow_error"):
        return "finalize"
    if state.get("research_report"):
        return "continue"
    if state.get("attempt_counts", {}).get(
        "research", 0
    ) < settings.agent_max_attempts.get("research", 3):
        return "retry"
    return "finalize"


def route_after_outline(state: WorkflowState) -> CreateContinueRoute:
    """Outline 没有产出页面结构时限次重试，耗尽后失败收尾。"""
    if state.get("workflow_error"):
        return "finalize"
    if state.get("outline"):
        return "continue"
    if state.get("attempt_counts", {}).get(
        "outline", 0
    ) < settings.agent_max_attempts.get("outline", 3):
        return "retry"
    return "finalize"


def route_after_content(state: WorkflowState) -> CreateContinueRoute:
    """Content 未同时产出文件名和页面清单时，未超上限则回指自身重试。"""
    if (
        state.get("filename")
        and state.get("slides_manifest")
        and not state.get("workflow_error")
    ):
        return "continue"
    if state.get("attempt_counts", {}).get(
        "content", 0
    ) < settings.agent_max_attempts.get("content", 3):
        return "retry"
    return "finalize"


def route_after_enhancement_planner(state: WorkflowState) -> PostPlannerRoute:
    """Planner 完成后按一次性增强计划进入 Assets、Beautify、重试或结束。

    requirements_initialized 尚未置位时表示 Planner 还未成功：未超上限则
    回指自身重试，超限后 Planner 会自行返回"降级为空计划"，落到 finalize。
    """
    if state.get("workflow_error"):
        return "finalize"
    if not state.get("requirements_initialized"):
        return "retry"
    required = set(state.get("required_stages", []))
    if "assets" in required:
        return "assets"
    if "beautify" in required:
        return "beautify"
    return "finalize"


def route_after_beautify(state: WorkflowState) -> PostBeautifyRoute:
    """Beautify 未产出有效美化结果且未超上限时回指自身重试。"""
    if "beautify" in state.get("completed_stages", []) or state.get("workflow_error"):
        return "finalize"
    if state.get("attempt_counts", {}).get(
        "beautify", 0
    ) < settings.agent_max_attempts.get("beautify", 3):
        return "retry"
    return "finalize"


def route_after_assets(state: WorkflowState) -> PostAssetsRoute:
    """Assets 完成后按规划决定是否执行最终美化。"""
    return "beautify" if "beautify" in state.get("required_stages", []) else "finalize"


def _build_create_subgraph(
    llm: BaseChatModel,
    assets_subgraph,
    ppt_context_service: PptContextService | None = None,
):
    """组装 Research → Outline → Content → Plan 的确定性主干。

    提供 MySQL 持久化服务时，会在 Research、Outline、Content 与 Assets
    完成后插入增量 persist；失败路径仍统一走 Finalize 后的最终 persist。
    """
    builder = StateGraph(WorkflowState)
    builder.add_node(RESEARCH_NODE, build_specialist_node("research", llm))
    builder.add_node(OUTLINE_NODE, build_specialist_node("outline", llm))
    builder.add_node(
        ENHANCEMENT_PLANNER_NODE,
        build_enhancement_planner_node(llm),
    )
    builder.add_node(CONTENT_NODE, build_specialist_node("content", llm))
    builder.add_node(ASSETS_NODE, assets_subgraph)
    builder.add_node(BEAUTIFY_NODE, build_specialist_node("beautify", llm))
    builder.add_node(FINALIZE_CREATE_NODE, finalize_create_node)
    has_persist = ppt_context_service is not None
    if has_persist:
        builder.add_node(
            PERSIST_RESEARCH_MILESTONE_NODE,
            build_persist_node(ppt_context_service),
        )
        builder.add_node(
            PERSIST_OUTLINE_MILESTONE_NODE,
            build_persist_node(ppt_context_service),
        )
        builder.add_node(
            PERSIST_CONTENT_MILESTONE_NODE,
            build_persist_node(ppt_context_service),
        )
        builder.add_node(
            PERSIST_ASSETS_MILESTONE_NODE,
            build_persist_node(ppt_context_service),
        )
        builder.add_node(
            PERSIST_PPT_RECORD_NODE,
            build_persist_node(ppt_context_service),
        )

    builder.add_conditional_edges(
        START,
        route_create_entry,
        {
            "research": RESEARCH_NODE,
            "outline": OUTLINE_NODE,
            "content": CONTENT_NODE,
            "planner": ENHANCEMENT_PLANNER_NODE,
            "assets": ASSETS_NODE,
            "beautify": BEAUTIFY_NODE,
            "finalize": FINALIZE_CREATE_NODE,
        },
    )
    builder.add_conditional_edges(
        RESEARCH_NODE,
        route_after_research,
        {
            "retry": RESEARCH_NODE,
            "continue": (
                PERSIST_RESEARCH_MILESTONE_NODE if has_persist else OUTLINE_NODE
            ),
            "finalize": FINALIZE_CREATE_NODE,
        },
    )
    if has_persist:
        builder.add_edge(PERSIST_RESEARCH_MILESTONE_NODE, OUTLINE_NODE)

    builder.add_conditional_edges(
        OUTLINE_NODE,
        route_after_outline,
        {
            "retry": OUTLINE_NODE,
            "continue": (
                PERSIST_OUTLINE_MILESTONE_NODE if has_persist else CONTENT_NODE
            ),
            "finalize": FINALIZE_CREATE_NODE,
        },
    )
    if has_persist:
        builder.add_edge(PERSIST_OUTLINE_MILESTONE_NODE, CONTENT_NODE)

    builder.add_conditional_edges(
        CONTENT_NODE,
        route_after_content,
        {
            "retry": CONTENT_NODE,
            "continue": (
                PERSIST_CONTENT_MILESTONE_NODE
                if has_persist
                else ENHANCEMENT_PLANNER_NODE
            ),
            "finalize": FINALIZE_CREATE_NODE,
        },
    )
    if has_persist:
        builder.add_edge(PERSIST_CONTENT_MILESTONE_NODE, ENHANCEMENT_PLANNER_NODE)

    builder.add_conditional_edges(
        ENHANCEMENT_PLANNER_NODE,
        route_after_enhancement_planner,
        {
            "retry": ENHANCEMENT_PLANNER_NODE,
            "assets": ASSETS_NODE,
            "beautify": BEAUTIFY_NODE,
            "finalize": FINALIZE_CREATE_NODE,
        },
    )
    if has_persist:
        # 资源写入完成后先落一次 in_progress（此时 slides_manifest 已包含
        # 图片/图表页），再按规划决定是否执行 Beautify。
        builder.add_edge(ASSETS_NODE, PERSIST_ASSETS_MILESTONE_NODE)
        builder.add_conditional_edges(
            PERSIST_ASSETS_MILESTONE_NODE,
            route_after_assets,
            {"beautify": BEAUTIFY_NODE, "finalize": FINALIZE_CREATE_NODE},
        )
    else:
        builder.add_conditional_edges(
            ASSETS_NODE,
            route_after_assets,
            {"beautify": BEAUTIFY_NODE, "finalize": FINALIZE_CREATE_NODE},
        )
    builder.add_conditional_edges(
        BEAUTIFY_NODE,
        route_after_beautify,
        {"retry": BEAUTIFY_NODE, "finalize": FINALIZE_CREATE_NODE},
    )
    # Finalize 只负责校验最小产物并写 FINISH 终态；真正落库在随后的最终
    # persist：有 workflow_error 记为 failed，否则 create_finalized 记为 completed。
    # 该校验并非无效：Content 等阶段工具静默失败（无异常但无产物）时，
    # 正是这里的 missing 检查兜底生成 workflow_error。
    builder.add_edge(
        FINALIZE_CREATE_NODE,
        PERSIST_PPT_RECORD_NODE if has_persist else END,
    )
    if has_persist:
        builder.add_edge(PERSIST_PPT_RECORD_NODE, END)
    return builder.compile()


def build_ppt_creation_subgraph(
    llm: BaseChatModel,
    ppt_context_service: PptContextService | None = None,
):
    """构建生产 Create 子图，并在结束前持久化最终 PPT 记录。"""
    return _build_create_subgraph(
        llm,
        build_assets_subgraph(llm),
        ppt_context_service,
    )


def build_debug_ppt_creation_subgraph(
    llm: BaseChatModel,
    ppt_context_service: PptContextService | None = None,
):
    """Create 路由不变，只替换为危险的 Debug Assets 子图。"""
    return _build_create_subgraph(
        llm,
        build_debug_assets_subgraph(llm),
        ppt_context_service,
    )
