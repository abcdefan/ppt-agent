"""Edit 暂时保留的 Supervisor 动态调度子图。"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.nodes.edit.supervisor import (
    EDIT_SUPERVISOR_NODE,
    build_edit_supervisor_node,
    route_after_edit_supervisor,
)
from app.agents.workflow.nodes.ppt_context.persist import (
    PERSIST_PPT_RECORD_NODE,
    build_persist_node,
    route_after_persist,
)
from app.agents.workflow.nodes.specialist import (
    BEAUTIFY_NODE,
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
    """构建当前只支持 Assets/Beautify 的动态 Edit Workflow。"""
    builder = StateGraph(WorkflowState)
    builder.add_node(EDIT_SUPERVISOR_NODE, build_edit_supervisor_node(llm))
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
            "assets": ASSETS_NODE,
            "beautify": BEAUTIFY_NODE,
            "FINISH": PERSIST_PPT_RECORD_NODE,
        },
    )
    builder.add_edge(ASSETS_NODE, PERSIST_PPT_RECORD_NODE)
    builder.add_edge(BEAUTIFY_NODE, PERSIST_PPT_RECORD_NODE)
    builder.add_conditional_edges(
        PERSIST_PPT_RECORD_NODE,
        route_after_persist,
        {"continue": EDIT_SUPERVISOR_NODE, "finish": END},
    )
    return builder.compile()
