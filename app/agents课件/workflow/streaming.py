"""workflow 模式流式映射：graph.astream_events(version="v2") → SSE 业务事件"""

import json
import logging
import os
from typing import AsyncIterator, Awaitable, Callable, Optional

from app.agents.common.tools import ROLES
from app.agents.workflow.router import CHAT_REPLY, INTENT_ROUTER
from app.schemas.events import (
    AGENT_SWITCH,
    TEXT_DELTA,
    TOOL_CALL,
    TOOL_RESULT,
    make_event,
)

logger = logging.getLogger(__name__)

# SSE 工具结果展示的最大长度（超出截断，避免单个事件过大）
TOOL_RESULT_PREVIEW_LEN = 500

# 顶层图节点名 — 用于 AGENT_SWITCH（create_agent 内部子链路名不会与之重名）
# 从 ROLES 派生，新增 specialist 时自动覆盖，无需手动维护
ROLE_NODES = {"supervisor", INTENT_ROUTER, CHAT_REPLY, *ROLES}


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_LEN) -> str:
    """截断过长的工具结果，供 SSE 事件展示"""
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + "...(已截断)"


async def stream_events_to_sse(
    graph,
    init_state: dict,
    recursion_limit: int,
    on_ppt_created: Optional[Callable[[str], Awaitable[None]]] = None,
) -> AsyncIterator[dict]:
    """订阅 langgraph 事件流，映射为 SSE 业务事件。

    事件映射：
    - AGENT_SWITCH : on_chain_start 且 name 为顶层节点 → 节点切换时发一次
    - TEXT_DELTA   : on_chat_model_stream → 逐 token（supervisor 节点的路由 JSON 过滤）
    - TOOL_CALL    : on_tool_start
    - TOOL_RESULT  : on_tool_end（截断）；generate_ppt 成功时触发 on_ppt_created 回调
    """
    current_node: Optional[str] = None
    config = {"recursion_limit": recursion_limit}

    async for ev in graph.astream_events(init_state, version="v2", config=config):
        kind = ev.get("event")
        name = ev.get("name", "")
        data = ev.get("data", {}) or {}

        # AGENT_SWITCH：顶层专家节点切换
        if kind == "on_chain_start" and name in ROLE_NODES:
            if name != current_node:
                current_node = name
                yield make_event(AGENT_SWITCH, {"agent": name})

        # 逐 token 文本
        elif kind == "on_chat_model_stream":
            # supervisor 的路由 JSON、intent_router 的分类 JSON 都不推给前端避免噪声；
            # chat_reply 的 token 才是给用户的回复，正常推送
            if current_node in ("supervisor", INTENT_ROUTER):
                continue
            chunk = data.get("chunk")
            content = getattr(chunk, "content", "") if chunk else ""
            if content:
                yield make_event(TEXT_DELTA, {"content": content})

        # 工具调用开始
        elif kind == "on_tool_start":
            tool_input = data.get("input")
            yield make_event(
                TOOL_CALL,
                {
                    "tool": name,
                    "args": tool_input if isinstance(tool_input, dict)
                    else {"input": tool_input},
                },
            )

        # 工具调用结束
        elif kind == "on_tool_end":
            output = data.get("output")
            content = getattr(output, "content", None)
            result_str = content if content else (
                str(output) if output is not None else ""
            )
            yield make_event(
                TOOL_RESULT, {"tool": name, "result": _truncate(result_str)}
            )

            # generate_ppt 成功 → 触发配额回调（控制器侧完成扣减 + 记录）
            if name == "generate_ppt" and on_ppt_created is not None:
                try:
                    result_obj = json.loads(result_str)
                    if (
                        isinstance(result_obj, dict)
                        and result_obj.get("success") is True
                    ):
                        file_path = result_obj.get("file_path", "")
                        filename = (
                            os.path.basename(file_path) or "presentation.pptx"
                        )
                        await on_ppt_created(filename)
                except (json.JSONDecodeError, TypeError) as e:
                    # 结果解析失败不致命（文件已落盘），仅记录
                    logger.warning(
                        "[workflow/流式] generate_ppt 结果解析失败: %s", e
                    )
