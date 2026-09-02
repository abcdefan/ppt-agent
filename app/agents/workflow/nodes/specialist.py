"""将 Specialist Agents 包装成 Workflow Graph Nodes。

本模块只完成最核心的适配：Node 从 State 构造专家任务，调用 Specialist
Agent，再把大纲或文件名等关键结果写回 State。节点注册和连线由后续的
``graph.py`` 负责。
"""

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agents.specialists.agent_registry import build_specialist_agent
from app.agents.specialists.research_agent import (
    normalize_research_report,
    validate_research_report,
)
from app.agents.specialists.tool_registry import AGENT_ROLES, AgentRole
from app.agents.workflow.state import WorkflowState
from app.core.config import settings

# Graph Node 接收当前 State，并返回需要合并回 State 的字段。
SpecialistNode = Callable[[WorkflowState], Awaitable[dict[str, Any]]]

logger = logging.getLogger(__name__)

RESEARCH_NODE = "research_node"
OUTLINE_NODE = "outline_node"
CONTENT_NODE = "content_node"
IMAGE_NODE = "image_node"
CHART_NODE = "chart_node"
BEAUTIFY_NODE = "beautify_node"

RETRYABLE_SPECIALIST_ROLES = {"research", "outline", "content", "beautify"}


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
        is_edit = state.get("intent") == "edit"
        # Edit 的节点失败直接结束本轮 Run，不使用 Create 的节点级重试或降级。
        retryable = agent_role in RETRYABLE_SPECIALIST_ROLES and not is_edit
        attempts = (
            state.get("attempt_counts", {}).get(agent_role, 0) + 1
            if retryable
            else 0
        )
        max_attempts = settings.agent_max_attempts.get(agent_role, 3)
        try:
            result = await specialist_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=task),
                    ]
                },
                config={
                    # 限制单次 Specialist Agent 内部的 ReAct 图步数，避免模型
                    # 持续在“思考 -> 调用工具 -> 再思考”之间循环。它与外层
                    # Workflow 的节点重试次数 agent_max_attempts 相互独立。
                    "recursion_limit": settings.agent_max_react_iterations,
                },
            )
        except Exception as exc:
            logger.exception("Specialist 执行失败: role=%s", agent_role)
            patch: dict[str, Any] = {}
            if retryable:
                patch["attempt_counts"] = {agent_role: attempts}
            # Image/Chart 可能处于并行分支，不写同一个标量错误字段，避免
            # LangGraph 并发更新冲突；Assets 收尾会根据操作结果判断状态。
            if retryable:
                # 可重试失败：只记录原因，不直接写死 workflow_error
                # （否则所有路由都会因 workflow_error 短路到 finalize，重试失效）。
                # 由对应路由函数判断是回指自身重试还是失败收尾。
                patch["attempt_error"] = f"{agent_role} 阶段执行失败：{exc}"
                if agent_role == "research" and attempts >= max_attempts:
                    patch.update(
                        {
                            "research_report": normalize_research_report(""),
                            "completed_agents": ["research"],
                            "completed_stages": ["research"],
                            "attempt_error": None,
                            "route_reason": "调研专家连续执行失败，已降级为空报告继续",
                        }
                    )
                elif agent_role == "beautify" and attempts >= max_attempts:
                    patch.update(
                        {
                            "required_stages": [
                                stage
                                for stage in state.get("required_stages", [])
                                if stage != "beautify"
                            ],
                            "attempt_error": None,
                            "route_reason": "美化连续执行失败，已降级交付未美化版本",
                        }
                    )
            elif agent_role not in {"image", "chart"}:
                patch["workflow_error"] = f"{agent_role} 阶段执行失败：{exc}"
            return patch

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

        # result_messages 只在当前 Specialist Node 内部使用。这里从 ReAct
        # 轨迹中提取可靠业务结果，绝不把 Tool Call/Tool Result 写入父 State。
        result_messages = result.get("messages", [])

        patch: dict[str, Any] = {}
        if retryable:
            patch["attempt_counts"] = {agent_role: attempts}
        completed = False

        if prepare_assets and agent_role in {"image", "chart"}:
            operations = _extract_asset_operations(result_messages)
            if operations:
                patch["asset_operations"] = operations
                completed = True
            else:
                completed = False

        if not prepare_assets and agent_role == "image":
            completed = _has_successful_tool_result(result_messages, "add_image_slide")

        if not prepare_assets and agent_role == "chart":
            completed = _has_successful_tool_result(result_messages, "add_chart_slide")

        if agent_role == "beautify":
            completed = _has_successful_tool_result(result_messages, "enhance_ppt")

        # Outline 的最终文本写入 state["outline"]，供 Content 读取。
        if agent_role == "outline":
            outline = _extract_final_ai_text(result_messages)
            if outline:
                patch["outline"] = outline
                completed = True
            else:
                patch["attempt_error"] = "outline 阶段未产生有效页面大纲"

        # 合法的 completed/partial/unavailable 报告均可继续；模型输出结构非法时
        # 先做节点级重试，达到上限后才降级为 unavailable 报告。
        if agent_role == "research":
            research_text = _extract_final_ai_text(result_messages)
            research_report = validate_research_report(research_text)
            if research_report is not None:
                patch["research_report"] = research_report
                completed = True
            elif attempts >= max_attempts:
                patch["research_report"] = normalize_research_report(research_text)
                patch["route_reason"] = "调研报告连续格式错误，已降级为空报告继续"
                completed = True
            else:
                patch["attempt_error"] = "research 阶段未产生合法的结构化调研报告"

        # Content 调用 generate_ppt 后，把真实文件名和实际用于生成 PPT 的
        # 页面内容一起写回 State，供 Planner 与后续增强节点读取。
        if agent_role == "content":
            filename, slides_manifest = _extract_generated_ppt(result_messages)
            if filename and slides_manifest:
                patch["filename"] = filename
                patch["slides_manifest"] = slides_manifest
                completed = True
            else:
                patch["attempt_error"] = "content 阶段未生成有效 PPT 文件和页面清单"

        if agent_role == "beautify" and not completed:
            if attempts >= max_attempts:
                patch["required_stages"] = [
                    stage
                    for stage in state.get("required_stages", [])
                    if stage != "beautify"
                ]
                patch["attempt_error"] = None
                patch["route_reason"] = "美化重试耗尽，已降级交付未美化版本"
            else:
                patch["attempt_error"] = "beautify 阶段未产生有效美化结果"

        if completed:
            patch["completed_agents"] = [agent_role]
            if agent_role in {"outline", "research", "content", "beautify"}:
                patch["completed_stages"] = [agent_role]
            # 重试成功后清除残留的可重试错误，避免 Finalize 误用旧错误。
            if retryable:
                patch["attempt_error"] = None

        if is_edit and not completed and agent_role in RETRYABLE_SPECIALIST_ROLES:
            failure_reason = patch.pop(
                "attempt_error",
                f"{agent_role} 阶段未产生有效业务产物",
            )
            patch["workflow_error"] = failure_reason

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
    """从准备型 ToolMessage 中提取 ppt_writer_node 可以执行的操作。"""
    tool_names = {"prepare_image_operation", "prepare_chart_operation"}
    operations: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in tool_names:
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
        if not isinstance(result, dict):
            continue
        operation = result.get("operation")
        if result.get("success") is True and isinstance(operation, dict):
            operations.append(operation)
    return operations


def _has_successful_tool_result(messages: list, tool_name: str) -> bool:
    """只有目标工具明确返回 success=true 时才视为阶段完成。"""
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != tool_name:
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
        if isinstance(result, dict) and result.get("success") is True:
            return True
    return False


def _build_specialist_task(
    agent_role: AgentRole,
    state: WorkflowState,
) -> str:
    """从 State 取得当前专家需要的上下文并组成任务。"""
    history_context = _render_conversation_history(
        state.get("conversation_history", [])
    )
    base_context = (
        "用户此前在本 Session 中的需求与讨论：\n"
        f"{history_context}\n\n"
        f"用户本轮请求：{state['user_message']}\n"
        f"PPT 风格：{state['style']}"
    )

    if agent_role == "outline":
        research_report = state.get("research_report")
        return (
            f"{base_context}\n"
            f"调研专家的研究报告 JSON：\n{research_report}\n"
            "请严格以用户需求和调研报告为依据，规划完整的 PPT 页面大纲；"
            "不得补造报告中不存在的外部事实。"
        )

    if agent_role == "research":
        return (
            f"{base_context}\n"
            "当前尚未生成大纲。请先从用户原始需求中提炼 3-6 个关键研究问题，"
            "再完成联网检索，输出规定格式的研究报告 JSON，供大纲专家使用。"
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
        "image": "请结合页面大纲，为适合视觉表达的页面准备配图操作。",
        "chart": (
            "请只使用调研报告中可靠的数值，为适合数据表达的页面准备图表操作；"
            "如果没有可用数值，不得编造。"
        ),
        "beautify": "请对这个 PPT 进行最终美化。",
        "outline": "",
        "research": "",
        "content": "",
    }
    enhancement_context = ""
    if agent_role in {"image", "chart"}:
        slides_manifest = json.dumps(
            state.get("slides_manifest") or [],
            ensure_ascii=False,
        )
        enhancement_context = (
            f"\n完整大纲：\n{state.get('outline') or '无可用大纲'}\n"
            f"当前 PPT 的实际页面内容：\n{slides_manifest}\n"
            f"调研专家的研究报告 JSON：\n"
            f"{state.get('research_report') or '无可用调研报告'}\n"
            f"增强规划理由：{state.get('route_reason') or '未提供'}\n"
        )

    return (
        f"{base_context}\n"
        f"当前 PPT 的真实文件名：{state['filename']}\n"
        f"{enhancement_context}"
        f"{action_by_role[agent_role]}"
    )


def _render_conversation_history(messages: list[BaseMessage]) -> str:
    """把入口加载的只读历史快照渲染成 Specialist Task 的背景信息。"""
    if not messages:
        return "（当前 Session 没有历史对话）"

    role_labels = {
        "system": "历史摘要",
        "human": "用户",
        "ai": "助手",
    }
    rendered: list[str] = []
    for message in messages:
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        if not content.strip():
            continue
        label = role_labels.get(message.type, message.type)
        rendered.append(f"[{label}] {content.strip()}")
    return "\n".join(rendered) or "（当前 Session 没有有效历史对话）"


def _extract_final_ai_text(messages: list) -> str:
    """提取 Specialist 最后一条不包含 Tool Call 的 AI 文本。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = str(message.text).strip()
            if text:
                return text
    return ""


def _extract_generated_ppt(
    messages: list,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """从 generate_ppt 的 ToolMessage 中提取文件名和实际页面内容。"""
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
            slides_manifest = result.get("slides_manifest")
            if (
                isinstance(file_path, str)
                and file_path
                and isinstance(slides_manifest, list)
                and slides_manifest
                and all(isinstance(slide, dict) for slide in slides_manifest)
            ):
                return os.path.basename(file_path), slides_manifest

    return None, None
