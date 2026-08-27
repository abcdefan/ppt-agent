"""PPT 创建子图：生产版单点写入，Debug 版保留并行直接写入。"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.edit_node import EDIT_NODE, edit_node
from app.agents.workflow.specialist_nodes import build_specialist_node
from app.agents.workflow.state import WorkflowState

PPT_CREATION_SUBGRAPH = "ppt_creation_subgraph"
DEBUG_PPT_CREATION_SUBGRAPH = "debug_ppt_creation_subgraph"
DEBUG_JOIN = "debug_join"

CreationEntry = Literal["outline", "research", "content"]


def _route_creation_entry(state: WorkflowState) -> CreationEntry:
    """允许子图被独立调用或恢复时从已有业务状态选择起点。"""
    requested = state.get("next")
    if requested in {"outline", "research", "content"}:
        return requested
    if not state.get("outline"):
        return "outline"
    if not state.get("research_report"):
        return "research"
    return "content"


def _add_common_creation_prefix(
    builder: StateGraph,
    llm: BaseChatModel,
) -> None:
    builder.add_node("outline", build_specialist_node("outline", llm))
    builder.add_node("research", build_specialist_node("research", llm))
    builder.add_node("content", build_specialist_node("content", llm))
    builder.add_conditional_edges(
        START,
        _route_creation_entry,
        {
            "outline": "outline",
            "research": "research",
            "content": "content",
        },
    )
    builder.add_edge("outline", "research")
    builder.add_edge("research", "content")


def build_ppt_creation_subgraph(llm: BaseChatModel):
    """生产子图：Agent 并行准备资源，edit_node 统一原子写入。"""
    builder = StateGraph(WorkflowState)
    _add_common_creation_prefix(builder, llm)

    builder.add_node(
        "image",
        build_specialist_node("image", llm, prepare_assets=True),
    )
    builder.add_node(
        "chart",
        build_specialist_node("chart", llm, prepare_assets=True),
    )
    builder.add_node(EDIT_NODE, edit_node)
    builder.add_node("beautify", build_specialist_node("beautify", llm))

    # 固定的两个异构分支使用静态 fan-out；两者处于同一 superstep。
    builder.add_edge("content", "image")
    builder.add_edge("content", "chart")

    # LangGraph 会在 image/chart 都结束后只执行一次 edit_node。
    builder.add_edge("image", EDIT_NODE)
    builder.add_edge("chart", EDIT_NODE)
    builder.add_edge(EDIT_NODE, "beautify")
    builder.add_edge("beautify", END)
    return builder.compile()


def build_debug_ppt_creation_subgraph(llm: BaseChatModel):
    """实验子图：Image/Chart 保留直接并行覆盖同一 PPT 的危险行为。

    本子图不注册锁 Tools，也不接入默认 Workflow。它仅用于未来通过
    LangSmith 观察直接并行写文件的竞争、等待和副作用行为。
    """
    builder = StateGraph(WorkflowState)
    _add_common_creation_prefix(builder, llm)

    builder.add_node("image", build_specialist_node("image", llm))
    builder.add_node("chart", build_specialist_node("chart", llm))

    async def debug_join(_state: WorkflowState) -> dict:
        return {}

    builder.add_node(DEBUG_JOIN, debug_join)
    builder.add_node("beautify", build_specialist_node("beautify", llm))

    # 两个 Agent 继续调用原有 add_image_slide/add_chart_slide，都会直接
    # 打开并覆盖同一文件。这里故意不修复竞争，只把危险实验隔离出来。
    builder.add_edge("content", "image")
    builder.add_edge("content", "chart")
    builder.add_edge("image", DEBUG_JOIN)
    builder.add_edge("chart", DEBUG_JOIN)
    builder.add_edge(DEBUG_JOIN, "beautify")
    builder.add_edge("beautify", END)
    return builder.compile()
