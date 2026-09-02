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

from app.agents.workflow.nodes.ppt_writer import PPT_WRITER_NODE
from app.agents.workflow.nodes.ppt_context.resolve import MATCH_ACTIVE_PPT_NODE
from app.agents.workflow.nodes.create.planner import ENHANCEMENT_PLANNER_NODE
from app.agents.workflow.nodes.edit.supervisor import EDIT_SUPERVISOR_NODE
from app.agents.workflow.nodes.reply import REPLY_NODE
from app.agents.workflow.nodes.router import INTENT_ROUTER_NODE
from app.agents.workflow.nodes.specialist import (
    BEAUTIFY_NODE,
    CHART_NODE,
    CONTENT_NODE,
    IMAGE_NODE,
    OUTLINE_NODE,
    RESEARCH_NODE,
)
from app.agents.workflow.state import WorkflowState
from app.schemas.events import (
    AGENT_RESULT,
    AGENT_SWITCH,
    INPUT_REQUIRED,
    INTENT_ROUTED,
    TEXT_DELTA,
    TOOL_CALL,
    TOOL_RESULT,
    make_event,
)

logger = logging.getLogger(__name__)

TOOL_RESULT_PREVIEW_LEN = 500
TOP_LEVEL_NODES = {
    INTENT_ROUTER_NODE,
    MATCH_ACTIVE_PPT_NODE,
    REPLY_NODE,
    RESEARCH_NODE,
    OUTLINE_NODE,
    CONTENT_NODE,
    IMAGE_NODE,
    CHART_NODE,
    PPT_WRITER_NODE,
    BEAUTIFY_NODE,
    EDIT_SUPERVISOR_NODE,
    ENHANCEMENT_PLANNER_NODE,
}

NODE_TO_AGENT_ROLE = {
    RESEARCH_NODE: "research",
    OUTLINE_NODE: "outline",
    CONTENT_NODE: "content",
    IMAGE_NODE: "image",
    CHART_NODE: "chart",
    PPT_WRITER_NODE: "writer",
    BEAUTIFY_NODE: "beautify",
}

OBSERVABLE_WORKFLOW_NODES = set(NODE_TO_AGENT_ROLE)

AGENT_COMPLETION_MESSAGES = {
    "outline": "大纲专家已完成 PPT 页面结构规划",
    "research": "调研专家已完成本轮联网资料检索",
    "content": "内容专家已完成基础 PPT 内容生成",
    "image": "配图专家已完成本轮配图处理",
    "chart": "图表专家已完成本轮图表处理",
    "beautify": "美化专家已完成本轮视觉优化",
    "writer": "PPT Writer 已统一提交图片和图表操作",
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
    thread_id: str | None = None,
    on_ppt_created: Callable[[str], Awaitable[None]] | None = None,
) -> AsyncIterator[dict]:
    """将 Workflow 的 v2 事件流转换成前端业务事件。

    课件通过顶层 ``on_chain_start`` 识别当前 Graph Node，
    只过滤 Supervisor 路由 JSON，会展示 Specialist 的模型文字。
    本项目保留节点识别方式，但根据界面需求同时过滤
    Specialist 文字，只向前端推送角色切换、专家结果摘要
    和 Tool 事件。
    """
    current_node: str | None = None
    emitted_interrupt_ids: set[str] = set()
    config = {"recursion_limit": recursion_limit}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}

    async for raw_event in graph.astream_events(
        initial_state,
        version="v2",
        config=config,
        # exit 模式只在本次 Graph 结束（包括 interrupt 退出）时
        # 落 checkpoint，符合本项目不做中途故障恢复的取舍。
        durability="exit",
    ):
        event_kind = raw_event.get("event", "")
        event_name = raw_event.get("name", "")
        event_data = raw_event.get("data") or {}
        call_id = str(raw_event.get("run_id") or "")

        # LangGraph 的动态 interrupt 不会作为普通异常抛出，而是在顶层
        # on_chain_stream chunk 中携带 __interrupt__。转换成稳定的业务事件，
        # 让前端能直接展示候选 PPT。
        if event_kind == "on_chain_stream":
            chunk = event_data.get("chunk")
            interrupts = (
                chunk.get("__interrupt__") if isinstance(chunk, dict) else None
            )
            if interrupts:
                for interrupt_item in interrupts:
                    interrupt_id = str(getattr(interrupt_item, "id", ""))
                    if interrupt_id and interrupt_id in emitted_interrupt_ids:
                        continue
                    if interrupt_id:
                        emitted_interrupt_ids.add(interrupt_id)
                    value = getattr(interrupt_item, "value", None)
                    if isinstance(value, dict):
                        yield make_event(INPUT_REQUIRED, value)
                continue

        # 顶层 Node 开始时，记录当前正在运行谁。Supervisor 和
        # Enhancement Planner 都是内部规划节点，不向前端展示结构化 JSON。
        if event_kind == "on_chain_start" and event_name in TOP_LEVEL_NODES:
            if event_name != current_node:
                current_node = event_name
                if event_name in OBSERVABLE_WORKFLOW_NODES:
                    yield make_event(
                        AGENT_SWITCH,
                        {"agent": NODE_TO_AGENT_ROLE[event_name]},
                    )
            continue

        # 只放行 Reply Node 的模型文字；Router/Supervisor/Planner 的结构化 JSON
        # 和 Specialist 的内部协作内容都不发给前端。
        # - Supervisor 输出的是 {"next": ...} 路由 JSON；
        # - Specialist 输出的是大纲 JSON、工具调用前说明和最终回复。
        # 这些都属于 Workflow 内部协作上下文，不转成前端
        # TEXT_DELTA。Create 的确定性完成文案不调用 LLM，会在下面的
        # Reply Node on_chain_end 分支发送。
        if event_kind == "on_chat_model_stream":
            if current_node == REPLY_NODE:
                chunk = event_data.get("chunk")
                content = getattr(chunk, "content", "") if chunk else ""
                if isinstance(content, str) and content:
                    yield make_event(TEXT_DELTA, {"content": content})
            continue

        if event_kind == "on_chain_end" and event_name == INTENT_ROUTER_NODE:
            output = event_data.get("output")
            if isinstance(output, dict):
                yield make_event(
                    INTENT_ROUTED,
                    {
                        "intent": output.get("intent", "chat"),
                        "execute": output.get("execute", False),
                        "source": output.get("route_source", "fallback"),
                        "confidence": output.get("route_confidence"),
                        "reason": output.get("route_reason", ""),
                    },
                )
            continue

        # Chat 已通过 on_chat_model_stream 逐 Token 发送；Create/Edit 不调用 LLM，
        # 因此只在统一 Reply Node 结束时发送 complete 模式的确定性完成文案。
        if event_kind == "on_chain_end" and event_name == REPLY_NODE:
            output = event_data.get("output")
            if isinstance(output, dict):
                content = output.get("final_response")
                response_mode = output.get("final_response_mode")
                if (
                    response_mode == "complete"
                    and isinstance(content, str)
                    and content
                ):
                    yield make_event(TEXT_DELTA, {"content": content})
            continue

        # 顶层 Specialist Node 正常结束时，只推送一条由代码
        # 组装的结果摘要。这不会暴露原始大纲 JSON、Prompt 或
        # “我将调用……”等模型过程文字。
        if event_kind == "on_chain_end" and event_name in OBSERVABLE_WORKFLOW_NODES:
            output = event_data.get("output")
            agent_role = NODE_TO_AGENT_ROLE[event_name]
            yield make_event(
                AGENT_RESULT,
                {
                    "agent": agent_role,
                    "message": _agent_result_message(
                        agent_role,
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
