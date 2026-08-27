"""将 Specialist Agents 包装成 Workflow Graph Nodes。

本模块只完成最核心的适配：Node 从 State 构造专家任务，调用 Specialist
Agent，再把大纲或文件名等关键结果写回 State。节点注册和连线由后续的
``graph.py`` 负责。
"""

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.specialists.agent_registry import build_specialist_agent
from app.agents.specialists.research_agent import normalize_research_report
from app.agents.specialists.tool_registry import AGENT_ROLES, AgentRole
from app.agents.workflow.state import WorkflowState

# Graph Node 接收当前 State，并返回需要合并回 State 的字段。
SpecialistNode = Callable[[WorkflowState], Awaitable[dict[str, Any]]]


def build_specialist_node(
    agent_role: AgentRole,
    llm: BaseChatModel,
    *,
    prepare_assets: bool = False,
) -> SpecialistNode:
    """创建一个 Specialist Agent，并将它包装成 Workflow Node。"""
    # 构建 Graph 时创建一次 Agent；run_specialist 闭包会一直持有它。
    specialist_agent = build_specialist_agent(
        agent_role=agent_role,
        llm=llm,
        prepare_assets=prepare_assets,
    )

    async def run_specialist(state: WorkflowState) -> dict[str, Any]:
        """根据 State 调用专家，并返回 State 的局部更新。"""
        task = _build_specialist_task(agent_role, state)
        result = await specialist_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=task),
                ]
            }
        )

        # result 是 Specialist Agent 本次完整的运行结果字典，例如：
        # {
        #     "messages": [
        #         HumanMessage(content=task),
        #         AIMessage(tool_calls=[...]),
        #         ToolMessage(content="工具执行结果"),
        #         AIMessage(content="专家最终回复"),
        #     ]
        # }
        # Outline 通常不调用业务工具，因此可能只有 HumanMessage 和 AIMessage。

        # result_messages 只在当前 Specialist Node 内部使用。这里会从中提取
        # 可靠业务结果，但绝不能把 Tool Call/Tool Result 原样合并到 Workflow
        # 的公共 messages，否则 Supervisor 截取上下文时可能破坏工具消息配对。
        result_messages = result.get("messages", [])

        patch: dict[str, Any] = {}
        completed = agent_role not in {"outline", "research", "content"}

        if prepare_assets and agent_role in {"image", "chart"}:
            operations = _extract_asset_operations(result_messages)
            if operations:
                patch["asset_operations"] = operations
                completed = True
            else:
                completed = False

        # Outline 的最终文本写入 state["outline"]，供 Content 读取。
        if agent_role == "outline":
            outline = _extract_final_ai_text(result_messages)
            if outline:
                patch["outline"] = outline
                completed = True

        # Research 无论正常完成、部分完成还是联网不可用，都会写入一份合法的
        # 报告并标记本轮完成，避免 Supervisor 因无 Key 或格式错误反复重试。
        if agent_role == "research":
            research_text = _extract_final_ai_text(result_messages)
            patch["research_report"] = normalize_research_report(research_text)
            completed = True

        # Content 调用 generate_ppt 后，把真实文件名写入 state["filename"]，
        # 供 Image、Chart 和 Beautify 读取。
        if agent_role == "content":
            filename = _extract_generated_filename(result_messages)
            if filename:
                patch["filename"] = filename
                completed = True

        if completed:
            patch["completed_agents"] = [agent_role]

        if agent_role == "content" and patch.get("filename"):
            summary = f"content 专家已完成，基础 PPT 已生成：{patch['filename']}"
        elif agent_role == "research":
            summary = "research 专家已完成本轮联网调研"
        elif completed:
            summary = f"{agent_role} 专家已完成本轮任务"
        else:
            summary = f"{agent_role} 专家本轮执行结束，但未产生有效业务结果"

        # 公共消息只保留普通文本摘要，绝不暴露 Specialist 的私有 ToolMessage。
        patch["messages"] = [
            HumanMessage(content=summary, name=agent_role),
        ]

        return patch

    return run_specialist


def build_specialist_nodes(
    llm: BaseChatModel,
    *,
    prepare_assets: bool = False,
) -> dict[AgentRole, SpecialistNode]:
    """一次性创建全部 Specialist Agents 及其 Workflow Nodes。"""
    return {
        agent_role: build_specialist_node(
            agent_role=agent_role,
            llm=llm,
            prepare_assets=prepare_assets,
        )
        for agent_role in AGENT_ROLES
    }


def _extract_asset_operations(messages: list) -> list[dict[str, Any]]:
    """从准备型 ToolMessage 中提取 edit_node 可以执行的操作。"""
    tool_names = {"prepare_image_operation", "prepare_chart_operation"}
    operations: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in tool_names:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        operation = result.get("operation")
        if result.get("success") is True and isinstance(operation, dict):
            operations.append(operation)
    return operations


def _build_specialist_task(
    agent_role: AgentRole,
    state: WorkflowState,
) -> str:
    """从 State 取得当前专家需要的上下文并组成任务。"""
    base_context = f"用户原始需求：{state['user_message']}\nPPT 风格：{state['style']}"

    if agent_role == "outline":
        return f"{base_context}\n请规划完整的 PPT 页面大纲。"

    if agent_role == "research":
        return (
            f"{base_context}\n"
            f"完整大纲：\n{state['outline']}\n"
            "请针对用户目标和每页大纲完成联网调研，输出规定格式的研究报告 JSON。"
        )

    if agent_role == "content":
        research_report = state.get("research_report")
        research_context = (
            f"\n调研专家的研究报告 JSON：\n{research_report}\n"
            if research_report
            else "\n本轮未执行联网调研，不得自行编造外部数据或来源。\n"
        )
        return (
            f"{base_context}\n"
            f"完整大纲：\n{state['outline']}\n"
            f"{research_context}"
            "请根据大纲生成基础 PPTX 文件。"
        )

    action_by_role: dict[AgentRole, str] = {
        "image": "请为这个 PPT 添加合适的配图。",
        "chart": "请为这个 PPT 添加合适的数据图表。",
        "beautify": "请对这个 PPT 进行最终美化。",
        "outline": "",
        "research": "",
        "content": "",
    }
    return (
        f"{base_context}\n"
        f"当前 PPT 的真实文件名：{state['filename']}\n"
        f"{action_by_role[agent_role]}"
    )


def _extract_final_ai_text(messages: list) -> str:
    """提取 Specialist 最后一条不包含 Tool Call 的 AI 文本。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = str(message.text).strip()
            if text:
                return text
    return ""


def _extract_generated_filename(messages: list) -> str | None:
    """从 generate_ppt 的 ToolMessage 中提取真实文件名。"""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage) or message.name != "generate_ppt":
            continue

        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(result, dict) and result.get("success"):
            file_path = result.get("file_path")
            if isinstance(file_path, str) and file_path:
                return os.path.basename(file_path)

    return None
