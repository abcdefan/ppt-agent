"""Workflow Graph 的构建与编译。

这个文件只负责“把已经定义好的 Node 串成图”：

    START -> Intent Router -> {Chat: Reply | Create: PPT Subgraph -> Reply} -> END

父图只负责意图分流和统一回复；PPT 专家与具体执行顺序全部封装在子图中。
"""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents.workflow.reply_node import REPLY, build_reply_node
from app.agents.workflow.router_node import INTENT_ROUTER, build_intent_router_node
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.subgraphs import (
    DEBUG_PPT_CREATION_SUBGRAPH,
    PPT_CREATION_SUBGRAPH,
    build_debug_ppt_creation_subgraph,
    build_ppt_creation_subgraph,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


CREATE_ROUTE = "create"


def route_after_intent(state: WorkflowState) -> str:
    """普通对话直接回复；创建任务直接进入 PPT Workflow 子图。"""
    return REPLY if state.get("intent") == "chat" else CREATE_ROUTE


def build_workflow_graph(llm: BaseChatModel, intent_router, chat_responder):
    """创建只负责 Intent 分流、PPT 子图挂载与统一回复的父图。"""
    # 1. 定义整张图共享的 State 结构。
    graph_builder = StateGraph(WorkflowState)

    # 2. 创建入口路由、统一回复和可切换的 PPT Workflow 子图。
    graph_builder.add_node(INTENT_ROUTER, build_intent_router_node(intent_router))
    graph_builder.add_node(REPLY, build_reply_node(chat_responder))
    if settings.agent_ppt_subgraph_mode == "debug":
        creation_subgraph_name = DEBUG_PPT_CREATION_SUBGRAPH
        creation_subgraph = build_debug_ppt_creation_subgraph(llm)
        logger.warning(
            "启用危险 Debug PPT 子图：Image/Chart 会直接并行写同一文件"
        )
    else:
        creation_subgraph_name = PPT_CREATION_SUBGRAPH
        creation_subgraph = build_ppt_creation_subgraph(llm)

    # 3. 父图只把编译后的子图当成一个 Node，不注册任何 Specialist。
    graph_builder.add_node(creation_subgraph_name, creation_subgraph)

    # 4. 整张图先做意图识别。
    graph_builder.add_edge(START, INTENT_ROUTER)
    graph_builder.add_conditional_edges(
        INTENT_ROUTER,
        route_after_intent,
        {
            REPLY: REPLY,
            CREATE_ROUTE: creation_subgraph_name,
        },
    )

    # 5. Chat 与 Create 最终汇入同一个 Reply Node。
    graph_builder.add_edge(REPLY, END)
    graph_builder.add_edge(creation_subgraph_name, REPLY)

    # compile() 之后得到真正可以 ainvoke()/astream_events() 的图。
    graph = graph_builder.compile()
    logger.info(
        "Workflow 父图构建完成: %s -> %s -> {%s | %s} -> %s",
        START,
        INTENT_ROUTER,
        REPLY,
        creation_subgraph_name,
        END,
    )
    return graph
