"""Workflow 原始运行事件 -> 项目现有 SSE 业务事件。

本文件不产生 SSE 字符串。它只订阅 ``graph.astream_events()``，
把 LangGraph/LangChain 的底层事件转换成 ``make_event()`` 字典；
Controller 会继续使用 ``app.core.sse.sse_event()`` 完成最后的 SSE 编码。
"""

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from app.agents.specialists.tool_registry import AGENT_ROLES
from app.agents.workflow.reply_node import CREATE_REPLY_MARKER, REPLY
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.edit_node import EDIT_NODE
from app.agents.workflow.router_node import INTENT_ROUTER
from app.schemas.events import (
    AGENT_RESULT,
    AGENT_SWITCH,
    INTENT_ROUTED,
    TEXT_DELTA,
    TOOL_CALL,
    TOOL_RESULT,
    make_event,
)

logger = logging.getLogger(__name__)

TOOL_RESULT_PREVIEW_LEN = 500
TOP_LEVEL_NODES = {
    INTENT_ROUTER,
    REPLY,
    *AGENT_ROLES,
    EDIT_NODE,
}

OBSERVABLE_WORKFLOW_ROLES = {*AGENT_ROLES, EDIT_NODE}

AGENT_COMPLETION_MESSAGES = {
    "outline": "大纲专家已完成 PPT 页面结构规划",
    "research": "调研专家已完成本轮联网资料检索",
    "content": "内容专家已完成基础 PPT 内容生成",
    "image": "配图专家已完成本轮配图处理",
    "chart": "图表专家已完成本轮图表处理",
    "beautify": "美化专家已完成本轮视觉优化",
    EDIT_NODE: "资源写入节点已统一提交图片和图表操作",
}


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_LEN) -> str:
    """截断过长的工具结果，避免单条 SSE 过大。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(已截断)"


def _tool_output_text(output: Any) -> str:
    """将 ToolMessage 或其他 Tool 返回值统一转成文本。"""
    if output is None:
        return ""
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        return str(content)
    return str(output)


def _tool_result_status(output: Any, result_text: str) -> str:
    """生成前端已经在使用的 success/error 状态。"""
    if getattr(output, "status", None) == "error":
        return "error"

    try:
        result_object = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        result_object = None

    if isinstance(result_object, dict) and result_object.get("success") is False:
        return "error"
    return "success"


def _agent_result_message(agent_role: str, output: Any) -> str:
    """根据 Node 返回的 State patch 生成简短、可展示的结果摘要。"""
    # Content Node 成功调用 generate_ppt 后会在 patch 中写入
    # filename。这是可以安全展示的业务结果，不是模型思考。
    if agent_role == "content" and isinstance(output, dict):
        filename = output.get("filename")
        if isinstance(filename, str) and filename:
            return f"内容专家已生成基础 PPT：{filename}"

    return AGENT_COMPLETION_MESSAGES[agent_role]


async def stream_workflow_events(
    graph,
    initial_state: WorkflowState,
    recursion_limit: int,
    on_ppt_created: Callable[[str], Awaitable[None]] | None = None,
    on_state_updated: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> AsyncIterator[dict]:
    """将 Workflow 的 v2 事件流转换成前端业务事件。

    课件通过顶层 ``on_chain_start`` 识别当前 Graph Node，
    只过滤 Supervisor 路由 JSON，会展示 Specialist 的模型文字。
    本项目保留节点识别方式，但根据界面需求同时过滤
    Specialist 文字，只向前端推送角色切换、专家结果摘要
    和 Tool 事件。
    """
    current_node: str | None = None
    config = {"recursion_limit": recursion_limit}

    async for raw_event in graph.astream_events(
        initial_state,
        version="v2",
        config=config,
    ):
        event_kind = raw_event.get("event", "")
        event_name = raw_event.get("name", "")
        event_data = raw_event.get("data") or {}
        call_id = str(raw_event.get("run_id") or "")

        # 顶层 Node 开始时，记录当前正在运行谁。Supervisor 只是
        # 内部路由节点；前端只需展示 Specialist 的角色切换。
        if event_kind == "on_chain_start" and event_name in TOP_LEVEL_NODES:
            if event_name != current_node:
                current_node = event_name
                if event_name in OBSERVABLE_WORKFLOW_ROLES:
                    yield make_event(AGENT_SWITCH, {"agent": event_name})
            continue

        # 只放行 Reply Node 的模型文字；Router/Supervisor 的结构化 JSON
        # 和 Specialist 的内部协作内容都不发给前端。
        # - Supervisor 输出的是 {"next": ...} 路由 JSON；
        # - Specialist 输出的是大纲 JSON、工具调用前说明和最终回复。
        # 这些都属于 Workflow 内部协作上下文，不转成前端
        # TEXT_DELTA。Create 的确定性完成文案不调用 LLM，会在下面的
        # Reply Node on_chain_end 分支发送。
        if event_kind == "on_chat_model_stream":
            if current_node == REPLY:
                chunk = event_data.get("chunk")
                content = getattr(chunk, "content", "") if chunk else ""
                if isinstance(content, str) and content:
                    yield make_event(TEXT_DELTA, {"content": content})
            continue

        if event_kind == "on_chain_end" and event_name == INTENT_ROUTER:
            output = event_data.get("output")
            if isinstance(output, dict):
                yield make_event(
                    INTENT_ROUTED,
                    {
                        "intent": output.get("intent", "chat"),
                        "source": output.get("route_source", "fallback"),
                        "confidence": output.get("route_confidence"),
                        "reason": output.get("route_reason", ""),
                    },
                )
            continue

        # Chat 已通过 on_chat_model_stream 逐 Token 发送；Create 不调用 LLM，
        # 因此只在统一 Reply Node 结束时发送带标记的确定性完成文案。
        if event_kind == "on_chain_end" and event_name == REPLY:
            output = event_data.get("output")
            messages = output.get("messages", []) if isinstance(output, dict) else []
            for message in reversed(messages):
                content = getattr(message, "content", "")
                reply_kind = getattr(message, "additional_kwargs", {}).get(
                    "workflow_reply_kind"
                )
                if (
                    reply_kind == CREATE_REPLY_MARKER
                    and isinstance(content, str)
                    and content
                ):
                    yield make_event(TEXT_DELTA, {"content": content})
                    break
            continue

        # 顶层 Specialist Node 正常结束时，只推送一条由代码
        # 组装的结果摘要。这不会暴露原始大纲 JSON、Prompt 或
        # “我将调用……”等模型过程文字。
        if event_kind == "on_chain_end" and event_name in OBSERVABLE_WORKFLOW_ROLES:
            output = event_data.get("output")
            if on_state_updated is not None and isinstance(output, dict):
                state_patch: dict[str, Any] = {}
                if isinstance(output.get("outline"), str):
                    state_patch["outline"] = output["outline"]
                if isinstance(output.get("filename"), str):
                    state_patch["active_ppt_filename"] = output["filename"]
                if state_patch:
                    await on_state_updated(state_patch)
            yield make_event(
                AGENT_RESULT,
                {
                    "agent": event_name,
                    "message": _agent_result_message(
                        event_name,
                        output,
                    ),
                },
            )
            continue

        if event_kind == "on_tool_start":
            tool_input = event_data.get("input")
            yield make_event(
                TOOL_CALL,
                {
                    "tool": event_name,
                    "args": (
                        tool_input
                        if isinstance(tool_input, dict)
                        else {"input": tool_input}
                    ),
                    "call_id": call_id,
                },
            )
            continue

        if event_kind == "on_tool_end":
            output = event_data.get("output")
            result_text = _tool_output_text(output)
            yield make_event(
                TOOL_RESULT,
                {
                    "tool": event_name,
                    "result": _truncate(result_text),
                    "call_id": call_id,
                    "status": _tool_result_status(output, result_text),
                },
            )

            # Agent 层只通知 Controller “PPT 已创建”；扣减配额和记录
            # 数据库等业务仍然由 Controller 传入的回调完成。
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
                    logger.warning(
                        "[Workflow/流式] generate_ppt 结果解析失败: %s",
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
