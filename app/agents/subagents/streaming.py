"""将 Subagents 模式的 LangChain 事件转换为前端业务事件。

这个模块不负责创建 HTTP 连接，也不直接生成 SSE 文本。它只负责订阅
Master Agent 的 ``astream_events()``，识别 Master、Subagent 和底层业务
Tool 的运行边界，再输出统一的事件字典。Controller 最后会通过
``app.core.sse.sse_event()`` 把这些字典序列化成 SSE 消息。
"""

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from app.agents.specialists.tool_registry import AGENT_ROLES, AgentRole
from app.schemas.events import (
    AGENT_SWITCH,
    TEXT_DELTA,
    TOOL_CALL,
    TOOL_RESULT,
    make_event,
)

logger = logging.getLogger(__name__)

# 工具结果可能包含很长的 JSON 或文本。SSE 只展示预览，避免单条事件过大；
# 完整结果仍会留在 LangGraph 的消息状态里供 Master 继续推理。
TOOL_RESULT_PREVIEW_LEN = 500

# Delegation Tool 的名字由 delegation_tools.build_delegation_tool() 生成。
# Master 调用这些 Tool，含义不是执行普通业务函数，而是将任务切换给 Subagent。
SUBAGENT_TOOL_NAMES: dict[str, AgentRole] = {
    f"{agent_role}_agent_tool": agent_role for agent_role in AGENT_ROLES
}


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_LEN) -> str:
    """截断需要推送给前端展示的工具结果。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(已截断)"


def _tool_output_text(output: Any) -> str:
    """将 ToolMessage 或其他工具返回值统一转换为字符串。"""
    if output is None:
        return ""

    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        return str(content)
    return str(output)


def _tool_result_status(output: Any, result_text: str) -> str:
    """从 LangChain ToolMessage 或工具 JSON 中判断执行是否成功。"""
    message_status = getattr(output, "status", None)
    if message_status == "error":
        return "error"

    try:
        result_object = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        result_object = None

    if isinstance(result_object, dict) and result_object.get("success") is False:
        return "error"

    normalized = result_text.lstrip().lower()
    if normalized.startswith(("错误：", "错误:", "error:", "error：")):
        return "error"
    return "success"


async def stream_subagent_events(
    master,
    messages: list,
    recursion_limit: int,
    on_ppt_created: Callable[[str], Awaitable[None]] | None = None,
) -> AsyncIterator[dict]:
    """把 Master 的 LangChain 原始事件映射为项目业务事件。

    事件映射规则：

    - Master 调用 ``*_agent_tool``：产生 ``AGENT_SWITCH``；
    - Subagent 调用 read_file/generate_ppt 等业务 Tool：产生
      ``TOOL_CALL`` 和 ``TOOL_RESULT``；
    - Master 的流式文字：产生 ``TEXT_DELTA``；
    - Subagent 的中间模型输出不会推给用户，避免展示内部 ReAct 过程。

    本函数产出的是 ``{"event": ..., "data": ...}`` 业务字典，还不是最终
    的 SSE 文本。SSE 序列化由 Controller 层完成。
    """
    current_role: AgentRole | None = None
    config = {"recursion_limit": recursion_limit}

    # astream_events() 会把 Master、Delegation Tool、Subagent 及其业务 Tool
    # 的事件放进同一条异步事件流。version="v2" 提供统一的事件结构。
    async for raw_event in master.astream_events(
        {"messages": messages},
        version="v2",
        config=config,
    ):
        event_kind = raw_event.get("event", "")
        event_name = raw_event.get("name", "")
        event_data = raw_event.get("data") or {}
        call_id = str(raw_event.get("run_id") or "")

        # Master 开始调用 Delegation Tool，意味着进入对应的 Subagent。
        # 这里用 AGENT_SWITCH 表达角色切换，不再重复发送 TOOL_CALL。
        if event_kind == "on_tool_start" and event_name in SUBAGENT_TOOL_NAMES:
            current_role = SUBAGENT_TOOL_NAMES[event_name]
            yield make_event(AGENT_SWITCH, {"agent": current_role})
            continue

        # Delegation Tool 执行结束，意味着 Subagent 已经返回结果，执行权重新
        # 回到 Master。该结果会展示为一次 TOOL_RESULT，也会被 Master 用于
        # 判断接下来应该调用哪个专家。
        if event_kind == "on_tool_end" and event_name in SUBAGENT_TOOL_NAMES:
            current_role = None
            result_text = _tool_output_text(event_data.get("output"))
            yield make_event(
                TOOL_RESULT,
                {
                    "tool": event_name,
                    "result": _truncate(result_text),
                    "call_id": call_id,
                    "status": _tool_result_status(
                        event_data.get("output"), result_text
                    ),
                },
            )
            continue

        # current_role 不为空时，下面这些普通 Tool 就是当前 Subagent 调用的
        # 业务工具，例如 content_specialist 调用 generate_ppt。
        if event_kind == "on_tool_start":
            tool_input = event_data.get("input")
            yield make_event(
                TOOL_CALL,
                {
                    "tool": event_name,
                    "call_id": call_id,
                    "args": (
                        tool_input
                        if isinstance(tool_input, dict)
                        else {"input": tool_input}
                    ),
                },
            )
            continue

        if event_kind == "on_tool_end":
            result_text = _tool_output_text(event_data.get("output"))
            yield make_event(
                TOOL_RESULT,
                {
                    "tool": event_name,
                    "result": _truncate(result_text),
                    "call_id": call_id,
                    "status": _tool_result_status(
                        event_data.get("output"), result_text
                    ),
                },
            )

            # Agent 层只通知“PPT 已成功生成”；扣减配额、写数据库等业务操作
            # 仍由 Controller 传入的回调负责，避免 Agent 直接依赖数据库。
            if event_name == "generate_ppt" and on_ppt_created is not None:
                try:
                    result_object = json.loads(result_text)
                    if (
                        isinstance(result_object, dict)
                        and result_object.get("success") is True
                    ):
                        file_path = result_object.get("file_path", "")
                        filename = os.path.basename(file_path) or "presentation.pptx"
                        await on_ppt_created(filename)
                except (json.JSONDecodeError, TypeError) as exc:
                    # PPT 工具已经运行结束，无法解析展示结果不应中断整条流。
                    logger.warning(
                        "[Subagents/流式] generate_ppt 结果解析失败: %s",
                        exc,
                    )
            continue

        if event_kind == "on_tool_error":
            error_text = str(event_data.get("error") or "工具执行失败")
            yield make_event(
                TOOL_RESULT,
                {
                    "tool": event_name,
                    "result": _truncate(error_text),
                    "call_id": call_id,
                    "status": "error",
                },
            )
            continue

        # 同一条事件流也会包含 Subagent 内部的模型输出。只有 current_role
        # 为 None 时才代表 Master 正在向用户生成最终文字，因此只推送这部分。
        if event_kind == "on_chat_model_stream" and current_role is None:
            chunk = event_data.get("chunk")
            content = getattr(chunk, "content", "") if chunk else ""
            if isinstance(content, str) and content:
                yield make_event(TEXT_DELTA, {"content": content})
