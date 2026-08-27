"""workflow 模式：StateGraph 编排 — Supervisor 路由 + Specialist 循环，直到 FINISH。

specialist ReAct agent 构造见 app.agents（每个 specialist 一个文件）；
本文件负责 workflow 专属的节点编排：为 specialist 注入 filename/style 上下文、
content 成功后回写 filename，以及图的拓扑（supervisor ↔ specialist 循环）。
"""

import json
import logging
import os
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agents import build_specialist
from app.agents.common.llm import build_llm
from app.agents.common.tools import ROLES
from app.agents.workflow.state import AgentState
from app.agents.workflow.supervisor import SUPERVISOR, build_supervisor_node
from app.agents.workflow.router import (
    CHAT_REPLY,
    INTENT_ROUTER,
    build_chat_reply_node,
    build_intent_router_node,
)

logger = logging.getLogger(__name__)


def route_after_supervisor(state: AgentState) -> str:
    """条件边路由：supervisor 的 next 决定下一个 specialist，或结束。"""
    nxt = state.get("next") or "FINISH"
    if nxt == "FINISH":
        return END
    return nxt


def route_after_intent(state: AgentState) -> str:
    """条件边路由：intent_router 的 intent 决定走 chat 直答，还是进 supervisor。"""
    intent = state.get("intent") or "chat"
    if intent == "chat":
        return CHAT_REPLY
    # create / enhance / analyze 本轮都进 supervisor 串行循环
    return SUPERVISOR


# ============================================================
# Specialist ReAct 节点包装（workflow 专属编排）
# ============================================================

def build_specialist_node(role: str, llm):
    """构建一个 specialist 节点：底层用 app/agents 的 ReAct agent，外层注入上下文 + 回写 filename。"""
    react = build_specialist(role, llm)

    async def node(state: AgentState) -> dict:
        # 注入 filename/style/outline 上下文，让 specialist 知道当前状态
        messages = list(state["messages"])
        messages.append(HumanMessage(content=_build_context_note(state, role)))

        result = await react.ainvoke({"messages": messages})
        new_msgs = result.get("messages", [])
        patch: dict = {"messages": new_msgs}

        # outline 完成规划 → 回写 outline 结构 JSON，供 research/content 消费
        if role == "outline" and not state.get("outline"):
            outline_json = _extract_outline(new_msgs)
            if outline_json:
                patch["outline"] = outline_json
                logger.info("[outline] 检测到大纲已生成（%d 字符）", len(outline_json))

        # research 完成调研 → 回写 research 笔记 JSON，供 content 消费
        if role == "research" and not state.get("research"):
            research_json = _extract_research(new_msgs)
            if research_json:
                patch["research"] = research_json
                logger.info("[research] 检测到研究笔记已生成（%d 字符）", len(research_json))

        # content 成功生成 PPT → 回写 filename 贯穿后续专家
        if role == "content" and not state.get("filename"):
            fn = _extract_filename(new_msgs)
            if fn:
                patch["filename"] = fn
                logger.info("[content] 检测到 PPT 已生成: %s", fn)
        return patch

    return node


def _build_context_note(state: AgentState, role: str) -> str:
    """为 specialist 构造上下文提示（大纲 / 研究笔记 / 文件名 / 风格）。"""
    filename = state.get("filename")
    style = state.get("style") or "business"
    outline = state.get("outline")
    research = state.get("research")
    user_message = state.get("user_message", "")

    if role == "outline":
        return (
            f"[上下文] 主题风格：{style}。请规划幻灯片结构，"
            "直接输出大纲 JSON 数组（每项含 title/layout_hint/purpose），不要生成文件。"
        )
    if role == "research":
        if outline:
            return (
                f"[上下文] 主题风格：{style}。以下是大纲专家产出的结构 JSON，"
                f"请据此为各页主题检索最新事实/数据/案例，并输出研究笔记 JSON：\n{outline}"
            )
        return (
            f"[上下文] 主题风格：{style}。尚无大纲，请基于用户主题检索资料并输出研究笔记 JSON。"
            f"用户主题：{user_message}"
        )
    if role == "content":
        parts = [f"[上下文] 主题风格：{style}。"]
        if outline:
            parts.append(
                "以下是大纲专家产出的结构 JSON，请据此为每页填充要点与备注并生成 PPTX：\n"
                f"{outline}"
            )
        else:
            parts.append("未提供大纲，请自行规划结构并生成 PPT。")
        if research:
            parts.append(
                "以下是调研专家收集的研究笔记 JSON，填充要点时请参考其中的事实/数据/结论"
                "（不要在正文里展示 URL）：\n" + research
            )
        parts.append("filename 请自行取一个语义化名字。")
        return "\n".join(parts)
    if filename:
        return (
            f"[上下文] 当前要操作的 PPT 文件名是：{filename}"
            f"（所有工具调用都必须传这个 filename）。主题风格：{style}。"
        )
    return (
        f"[上下文] 注意：尚未获取到 PPT 文件名，请向协调员反馈需要先生成 PPT。"
        f"主题风格：{style}。"
    )


def _last_ai_text(messages) -> Optional[str]:
    """取最后一条非空 AIMessage 文本（outline/research 均以末条 AIMessage 输出 JSON）。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return None


def _extract_outline(messages) -> Optional[str]:
    """取 outline 专家最后一条 AIMessage 文本作为大纲结构 JSON。"""
    return _last_ai_text(messages)


def _extract_research(messages) -> Optional[str]:
    """取 research 专家最后一条 AIMessage 文本作为研究笔记 JSON。"""
    return _last_ai_text(messages)


def _extract_filename(messages) -> Optional[str]:
    """从消息流中找到 generate_ppt 的成功返回，提取 file_path 的 basename。"""
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", "") != "generate_ppt":
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("success") and data.get("file_path"):
            return os.path.basename(data["file_path"])
    return None


# ============================================================
# 图构建
# ============================================================

def build_graph(llm=None):
    """构建并编译 workflow 多智能体图。

    Args:
        llm: 共享的 ChatOpenAI 实例（不传则新建）。router/supervisor/specialist 复用同一 LLM。
    """
    if llm is None:
        llm = build_llm()

    graph = StateGraph(AgentState)
    # 入口意图路由 + chat 直答
    graph.add_node(INTENT_ROUTER, build_intent_router_node(llm))
    graph.add_node(CHAT_REPLY, build_chat_reply_node(llm))
    # 原有 supervisor + specialists
    graph.add_node(SUPERVISOR, build_supervisor_node(llm))
    for role in ROLES:
        graph.add_node(role, build_specialist_node(role, llm))

    # 入口 → intent_router
    graph.add_edge(START, INTENT_ROUTER)
    # intent_router → chat 直答短路 / 或 supervisor 串行循环
    graph.add_conditional_edges(
        INTENT_ROUTER,
        route_after_intent,
        {CHAT_REPLY: CHAT_REPLY, SUPERVISOR: SUPERVISOR},
    )
    # chat_reply → END（不再回 supervisor）
    graph.add_edge(CHAT_REPLY, END)

    # supervisor → 条件路由 → specialist 或 END（保持不变）
    graph.add_conditional_edges(
        SUPERVISOR,
        route_after_supervisor,
        {**{role: role for role in ROLES}, END: END},
    )
    # 每个 specialist 执行完回到 supervisor（循环直至 FINISH）
    for role in ROLES:
        graph.add_edge(role, SUPERVISOR)

    compiled = graph.compile()
    logger.info(
        "workflow 图构建完成: intent_router → {chat→END | supervisor → %s}", ROLES
    )
    return compiled
