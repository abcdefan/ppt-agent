"""subagents 模式流式映射：主 agent 的 astream_events → SSE 业务事件。

从原 agent.py 拆出。主 agent 调用子 agent 工具的边界用 AGENT_SWITCH 表示；
子 agent 内部 ReAct 思考过滤，仅推主 agent 文本。
"""

import json
import logging
import os
from typing import AsyncIterator, Awaitable, Callable, Optional

from app.agents.common.tools import ROLES
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

# 子 agent 工具名集合（主 agent 调用这些 = 切换到对应专家）
SUBAGENT_TOOL_NAMES = {f"{role}_agent" for role in ROLES}


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_LEN) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + "...(已截断)"


async def stream_subagent_events(
    master,
    messages: list,
    recursion_limit: int,
    on_ppt_created: Optional[Callable[[str], Awaitable[None]]] = None,
) -> AsyncIterator[dict]:
    """主 agent 事件流 → SSE。

    - AGENT_SWITCH  : 主 agent 调用子 agent 工具（on_tool_start, name=*_agent）→ 切换到该专家
    - TOOL_CALL/RESULT : 子 agent 内部的工具调用（refine_content/generate_ppt/...）
    - TEXT_DELTA    : 仅主 agent 的文本（current_role is None）；子 agent 内部 ReAct 思考过滤
    """
    current_role: Optional[str] = None  # None=主 agent；否则=当前子 agent 角色
    config = {"recursion_limit": recursion_limit}

    async for ev in master.astream_events({"messages": messages}, version="v2", config=config):
        kind = ev.get("event")
        name = ev.get("name", "")
        data = ev.get("data", {}) or {}

        # 主 agent 调用子 agent 工具 → 切换到该专家（用 AGENT_SWITCH 表示，不重复发 TOOL_CALL）
        if kind == "on_tool_start" and name in SUBAGENT_TOOL_NAMES:
            current_role = name[:-6]  # 去掉 "_agent"
            yield make_event(AGENT_SWITCH, {"agent": current_role})
            continue

        # 子 agent 工具返回 → 回到主 agent
        if kind == "on_tool_end" and name in SUBAGENT_TOOL_NAMES:
            current_role = None
            output = data.get("output")
            content = getattr(output, "content", None)
            result_str = content if content else (str(output) if output is not None else "")
            yield make_event(TOOL_RESULT, {"tool": name, "result": _truncate(result_str)})
            continue

        # 子 agent 内部的工具调用开始
        if kind == "on_tool_start":
            tool_input = data.get("input")
            yield make_event(
                TOOL_CALL,
                {
                    "tool": name,
                    "args": tool_input if isinstance(tool_input, dict)
                    else {"input": tool_input},
                },
            )

        # 子 agent 内部的工具调用结束
        elif kind == "on_tool_end":
            output = data.get("output")
            content = getattr(output, "content", None)
            result_str = content if content else (str(output) if output is not None else "")
            yield make_event(TOOL_RESULT, {"tool": name, "result": _truncate(result_str)})

            # generate_ppt 成功 → 触发配额回调
            if name == "generate_ppt" and on_ppt_created is not None:
                try:
                    result_obj = json.loads(result_str)
                    if isinstance(result_obj, dict) and result_obj.get("success") is True:
                        file_path = result_obj.get("file_path", "")
                        await on_ppt_created(
                            os.path.basename(file_path) or "presentation.pptx"
                        )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("[subagents/流式] generate_ppt 结果解析失败: %s", e)

        # 仅主 agent 的文本；过滤掉子 agent 内部思考
        elif kind == "on_chat_model_stream":
            if current_role is not None:
                continue
            chunk = data.get("chunk")
            content = getattr(chunk, "content", "") if chunk else ""
            if content:
                yield make_event(TEXT_DELTA, {"content": content})
