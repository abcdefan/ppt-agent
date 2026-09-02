"""Edit Supervisor 动态调度全部已有业务阶段的子图。"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.nodes.edit.supervisor import (
    EDIT_SUPERVISOR_NODE,
    build_edit_supervisor_node,
    route_after_edit_supervisor,
)
from app.agents.workflow.nodes.create.planner import (
    ENHANCEMENT_PLANNER_NODE,
    build_enhancement_planner_node,
)
from app.agents.workflow.nodes.ppt_context.persist import (
    PERSIST_PPT_RECORD_NODE,
    build_persist_node,
    route_after_persist,
)
from app.agents.workflow.nodes.specialist import (
    BEAUTIFY_NODE,
    CONTENT_NODE,
    OUTLINE_NODE,
    RESEARCH_NODE,
    build_specialist_node,
)
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.subgraphs.assets import ASSETS_NODE, build_assets_subgraph
from app.services import PptContextService

PPT_EDIT_SUBGRAPH = "ppt_edit_subgraph"


def build_ppt_edit_subgraph(
    llm: BaseChatModel,
    ppt_context_service: PptContextService,
):
    """构建由 Supervisor 每轮自由选择下一阶段的 Edit Workflow。"""
    builder = StateGraph(WorkflowState)
    builder.add_node(EDIT_SUPERVISOR_NODE, build_edit_supervisor_node(llm))
    builder.add_node(RESEARCH_NODE, build_specialist_node("research", llm))
    builder.add_node(OUTLINE_NODE, build_specialist_node("outline", llm))
    builder.add_node(CONTENT_NODE, build_specialist_node("content", llm))
    builder.add_node(
        ENHANCEMENT_PLANNER_NODE,
        build_enhancement_planner_node(llm),
    )
    builder.add_node(ASSETS_NODE, build_assets_subgraph(llm))
    builder.add_node(BEAUTIFY_NODE, build_specialist_node("beautify", llm))
    builder.add_node(
        PERSIST_PPT_RECORD_NODE,
        build_persist_node(ppt_context_service),
    )

    builder.add_edge(START, EDIT_SUPERVISOR_NODE)
    builder.add_conditional_edges(
        EDIT_SUPERVISOR_NODE,
        route_after_edit_supervisor,
        {
            "research": RESEARCH_NODE,
            "outline": OUTLINE_NODE,
            "content": CONTENT_NODE,
            "planner": ENHANCEMENT_PLANNER_NODE,
            "assets": ASSETS_NODE,
            "beautify": BEAUTIFY_NODE,
            "FINISH": PERSIST_PPT_RECORD_NODE,
        },
    )
    # 所有阶段完成后统一增量落库；普通成功回 Supervisor，失败或 FINISH 结束。
    for stage_node in (
        RESEARCH_NODE,
        OUTLINE_NODE,
        CONTENT_NODE,
        ENHANCEMENT_PLANNER_NODE,
        ASSETS_NODE,
        BEAUTIFY_NODE,
    ):
        builder.add_edge(stage_node, PERSIST_PPT_RECORD_NODE)
    builder.add_conditional_edges(
        PERSIST_PPT_RECORD_NODE,
        route_after_persist,
        {"continue": EDIT_SUPERVISOR_NODE, "finish": END},
    )
    return builder.compile()
