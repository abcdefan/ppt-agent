"""Image/Chart 条件并行、PPT Writer 单点写入的 Assets 子图。"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.nodes.ppt_writer import (
    ASSETS_SKIP_NODE,
    DEBUG_ASSETS_JOIN_NODE,
    PPT_WRITER_NODE,
    debug_assets_join_node,
    ppt_writer_node,
    skip_assets_node,
)
from app.agents.workflow.nodes.specialist import (
    CHART_NODE,
    IMAGE_NODE,
    build_specialist_node,
)
from app.agents.workflow.state import AssetTask, WorkflowState

AssetRoute = Literal["image", "chart", "skip"]
ASSETS_NODE = "assets_node"


def route_assets(state: WorkflowState) -> AssetRoute | list[AssetTask]:
    """根据 Enhancement Planner 冻结的任务选择单分支、并行或跳过。"""
    tasks = set(state.get("asset_tasks", []))
    selected = [task for task in ("image", "chart") if task in tasks]
    if len(selected) == 2:
        return selected
    if selected:
        return selected[0]
    return "skip"


def build_assets_subgraph(llm: BaseChatModel):
    """生产 Assets 子图：Image/Chart 只准备操作，PPT Writer 统一写文件。"""
    builder = StateGraph(WorkflowState)
    builder.add_node(
        IMAGE_NODE,
        build_specialist_node("image", llm, prepare_assets=True),
    )
    builder.add_node(
        CHART_NODE,
        build_specialist_node("chart", llm, prepare_assets=True),
    )
    builder.add_node(PPT_WRITER_NODE, ppt_writer_node)
    builder.add_node(ASSETS_SKIP_NODE, skip_assets_node)

    builder.add_conditional_edges(
        START,
        route_assets,
        {
            "image": IMAGE_NODE,
            "chart": CHART_NODE,
            "skip": ASSETS_SKIP_NODE,
        },
    )
    # 同时选择 Image/Chart 时，两条边在同一 superstep 汇合，Writer 只执行一次。
    builder.add_edge(IMAGE_NODE, PPT_WRITER_NODE)
    builder.add_edge(CHART_NODE, PPT_WRITER_NODE)
    builder.add_edge(PPT_WRITER_NODE, END)
    builder.add_edge(ASSETS_SKIP_NODE, END)
    return builder.compile()


def build_debug_assets_subgraph(llm: BaseChatModel):
    """实验子图：保留 Image/Chart 直接并行写文件的危险实现。"""
    builder = StateGraph(WorkflowState)
    builder.add_node(IMAGE_NODE, build_specialist_node("image", llm))
    builder.add_node(CHART_NODE, build_specialist_node("chart", llm))

    builder.add_node(DEBUG_ASSETS_JOIN_NODE, debug_assets_join_node)
    builder.add_node(ASSETS_SKIP_NODE, skip_assets_node)
    builder.add_conditional_edges(
        START,
        route_assets,
        {
            "image": IMAGE_NODE,
            "chart": CHART_NODE,
            "skip": ASSETS_SKIP_NODE,
        },
    )
    builder.add_edge(IMAGE_NODE, DEBUG_ASSETS_JOIN_NODE)
    builder.add_edge(CHART_NODE, DEBUG_ASSETS_JOIN_NODE)
    builder.add_edge(DEBUG_ASSETS_JOIN_NODE, END)
    builder.add_edge(ASSETS_SKIP_NODE, END)
    return builder.compile()
