"""Workflow Graph 的构建与编译。

这个文件只负责“把已经定义好的 Node 串成图”：

    START -> Intent Router -> {
        Chat | Create Init | Active PPT Match -> Edit Target
    } -> 对应 PPT 子图 -> Reply

父图只负责意图分流和统一回复；PPT 专家与具体执行顺序全部封装在子图中。
"""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.nodes.ppt_context.initialize import (
    INITIALIZE_PPT_NODE,
    build_initialize_node,
)
from app.agents.workflow.nodes.ppt_context.resolve import (
    MATCH_ACTIVE_PPT_NODE,
    RESOLVE_TARGET_PPT_NODE,
    build_active_ppt_match_node,
    build_resolve_node,
    route_after_resolution,
)
from app.agents.workflow.nodes.reply import REPLY_NODE, build_reply_node
from app.agents.workflow.nodes.router import (
    INTENT_ROUTER_NODE,
    build_intent_router_node,
)
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.subgraphs import (
    DEBUG_PPT_CREATION_SUBGRAPH,
    PPT_CREATION_SUBGRAPH,
    PPT_EDIT_SUBGRAPH,
    build_debug_ppt_creation_subgraph,
    build_ppt_creation_subgraph,
    build_ppt_edit_subgraph,
)
from app.context import PptRecordStore, SessionStateStore
from app.core.config import settings
from app.services import PptContextService

logger = logging.getLogger(__name__)


CREATE_ROUTE = "create"
EDIT_ROUTE = "edit"
TARGET_RESOLVED = "resolved"
TARGET_ERROR = "error"


def route_after_intent(state: WorkflowState) -> str:
    """未获执行授权时统一对话；仅执行态准备 Create/Edit 上下文。"""
    intent = state.get("intent")
    if not state.get("execute", False):
        return REPLY_NODE
    if intent == "create":
        return CREATE_ROUTE
    if intent == "edit":
        return EDIT_ROUTE
    return REPLY_NODE


def build_workflow_graph(
    llm: BaseChatModel,
    intent_router,
    chat_responder,
    session_store: SessionStateStore | None = None,
    ppt_record_store: PptRecordStore | None = None,
    ppt_context_service: PptContextService | None = None,
    creation_subgraph=None,
    target_match_llm: BaseChatModel | None = None,
    *,
    checkpointer: BaseCheckpointSaver,
):
    """创建只负责 Intent 分流、PPT 子图挂载与统一回复的父图。"""
    # 1. 定义整张图共享的 State 结构。
    graph_builder = StateGraph(WorkflowState)
    session_store = session_store or SessionStateStore()
    ppt_record_store = ppt_record_store or PptRecordStore()

    # 2. 创建入口路由、统一回复以及 Create/Edit 各自的子图。
    graph_builder.add_node(INTENT_ROUTER_NODE, build_intent_router_node(intent_router))
    graph_builder.add_node(REPLY_NODE, build_reply_node(chat_responder))
    graph_builder.add_node(
        INITIALIZE_PPT_NODE,
        build_initialize_node(
            session_store,
            ppt_record_store,
            ppt_context_service,
        ),
    )
    graph_builder.add_node(
        MATCH_ACTIVE_PPT_NODE,
        build_active_ppt_match_node(
            target_match_llm or llm,
            ppt_context_service,
        ),
    )
    graph_builder.add_node(
        RESOLVE_TARGET_PPT_NODE,
        build_resolve_node(
            session_store,
            ppt_record_store,
            ppt_context_service,
        ),
    )
    if settings.agent_ppt_subgraph_mode == "debug":
        creation_subgraph_name = DEBUG_PPT_CREATION_SUBGRAPH
        if creation_subgraph is None:
            creation_subgraph = build_debug_ppt_creation_subgraph(
                llm,
                ppt_context_service,
            )
        logger.warning("启用危险 Debug PPT 子图：Image/Chart 会直接并行写同一文件")
    else:
        creation_subgraph_name = PPT_CREATION_SUBGRAPH
        if creation_subgraph is None:
            creation_subgraph = build_ppt_creation_subgraph(
                llm,
                ppt_context_service,
            )
    edit_subgraph = build_ppt_edit_subgraph(
        llm,
        ppt_context_service,
    )

    # 3. 父图只把编译后的子图当成 Node，不注册任何 Specialist。
    graph_builder.add_node(creation_subgraph_name, creation_subgraph)
    graph_builder.add_node(PPT_EDIT_SUBGRAPH, edit_subgraph)

    # 4. 整张图先做意图识别。
    graph_builder.add_edge(START, INTENT_ROUTER_NODE)
    graph_builder.add_conditional_edges(
        INTENT_ROUTER_NODE,
        route_after_intent,
        {
            REPLY_NODE: REPLY_NODE,
            CREATE_ROUTE: INITIALIZE_PPT_NODE,
            EDIT_ROUTE: MATCH_ACTIVE_PPT_NODE,
        },
    )

    graph_builder.add_edge(INITIALIZE_PPT_NODE, creation_subgraph_name)
    # 先把 LLM 的 Active PPT 核验结果写入 State，再进入可能 interrupt 的
    # Resolve 节点。恢复时只重跑 Resolve，不会再次调用目标核验 LLM。
    graph_builder.add_edge(MATCH_ACTIVE_PPT_NODE, RESOLVE_TARGET_PPT_NODE)
    graph_builder.add_conditional_edges(
        RESOLVE_TARGET_PPT_NODE,
        route_after_resolution,
        {
            TARGET_RESOLVED: PPT_EDIT_SUBGRAPH,
            TARGET_ERROR: REPLY_NODE,
        },
    )

    # 5. Chat、Create 与 Edit 最终汇入同一个 Reply Node。
    graph_builder.add_edge(REPLY_NODE, END)
    graph_builder.add_edge(creation_subgraph_name, REPLY_NODE)
    graph_builder.add_edge(PPT_EDIT_SUBGRAPH, REPLY_NODE)

    # Checkpointer 只绑定在父图。挂载为 Node 的子图会沿用父图的
    # checkpoint 上下文，无需再为每个子图创建 Redis Saver。
    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info(
        "Workflow 父图构建完成: %s -> %s -> {%s | %s | %s} -> %s",
        START,
        INTENT_ROUTER_NODE,
        REPLY_NODE,
        creation_subgraph_name,
        PPT_EDIT_SUBGRAPH,
        END,
    )
    return graph
