"""subagents 模式入口 —— 主 agent（项目经理）通过工具调用委托子 agent。

从原 agent.py 拆出：
- make_subagent_tool 把 app/agents 的 specialist 包装成 @tool（主 agent 像调工具一样委托）
- SubAgents 入口类（run / run_stream / run_ppt_stream / clear_history），签名与 MultiAgent 一致

底层 specialist ReAct agent 构造见 app.agents；流式映射见本目录 streaming.py。

对外接口与 MultiAgent / 旧 agent 完全一致。
"""

import logging
import time
from typing import AsyncIterator, Awaitable, Callable, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool

from app.agents import build_specialist
from app.agents.common.llm import build_llm, build_summary_llm
from app.agents.common.tools import ROLES
from app.agents.subagents.prompts import MASTER_PROMPT, ROLE_TOOL_DESCRIPTIONS
from app.agents.subagents.streaming import stream_subagent_events
from app.core.config import settings
from app.schemas.events import (
    AGENT_THINKING,
    DONE,
    ERROR,
    TEXT_DELTA,
    make_event,
)
from app.utils.memory import SummaryBufferMemory

try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # 兜底
    from langgraph.graph.exc import GraphRecursionError  # type: ignore

logger = logging.getLogger(__name__)


def _last_ai_text(result: dict) -> str:
    """从 agent 结果取最后一条有内容的 AIMessage 文本。"""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "（智能体未返回文字）"


# ============================================================
# 子 agent 工具化 —— 把 app/agents 的 specialist 包装成主 agent 可调用的 @tool
# ============================================================
def make_subagent_tool(role: str, llm):
    """把一个 specialist（app/agents 的 ReAct agent）包装成主 agent 可调用的工具。

    主 agent 调用此工具时，内部跑一次该角色的 ReAct specialist，返回其最终文本。
    filename 等上下文由主 agent 写进 task 描述传入（subagents 模式的自然交互）。
    """
    specialist = build_specialist(role, llm)
    tool_name = f"{role}_agent"

    @lc_tool(tool_name, description=ROLE_TOOL_DESCRIPTIONS[role])
    async def call_specialist(task: str) -> str:
        f"""调用 {role} 专家完成子任务。task 为自然语言任务描述（含上下文）。"""
        result = await specialist.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": 25},
        )
        return _last_ai_text(result)

    return call_specialist


# ============================================================
# 入口类
# ============================================================
class SubAgents:
    """subagents 模式入口（主 agent + 子 agent 工具）。"""

    def __init__(self):
        self.llm = build_llm()
        self.memory = SummaryBufferMemory(summary_llm=build_summary_llm())
        self.sub_tools = [make_subagent_tool(r, self.llm) for r in ROLES]
        self.master = create_agent(
            model=self.llm,
            tools=self.sub_tools,
            system_prompt=MASTER_PROMPT,
        )
        self.recursion_limit = settings.agent_multi_recursion_limit
        logger.info("SubAgents 初始化完成（主 agent + 子 agent %s）", ROLES)

    async def run(self, user_message: str, session_id: Optional[str] = None) -> str:
        session_id = session_id or "default"
        logger.info("[SubAgents] 会话 %s 收到消息: %s", session_id, user_message[:100])
        history = await self.memory.load(session_id)
        messages = list(history) + [HumanMessage(content=user_message)]
        try:
            result = await self.master.ainvoke(
                {"messages": messages}, config={"recursion_limit": self.recursion_limit}
            )
        except GraphRecursionError:
            logger.error("[SubAgents] 超出最大推理步数")
            return "抱歉，任务过于复杂，已达到最大推理步数限制，请尝试简化需求。"
        reply = _last_ai_text(result)
        await self.memory.save(session_id, user_message, reply)
        return reply

    async def run_stream(
        self, user_message: str, session_id: Optional[str] = None
    ) -> AsyncIterator[dict]:
        async for event in self._stream(user_message, session_id, None):
            yield event

    async def run_ppt_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        on_ppt_created: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> AsyncIterator[dict]:
        async for event in self._stream(user_message, session_id, on_ppt_created):
            yield event

    async def _stream(
        self,
        user_message: str,
        session_id: Optional[str],
        on_ppt_created: Optional[Callable[[str], Awaitable[None]]],
    ) -> AsyncIterator[dict]:
        session_id = session_id or "default"
        start = time.time()
        response_parts: list[str] = []

        logger.info("[SubAgents/流式] 会话 %s 收到消息: %s", session_id, user_message[:100])
        yield make_event(AGENT_THINKING, {"message": "多智能体团队开始协作（Subagents 模式）..."})

        try:
            history = await self.memory.load(session_id)
            messages = list(history) + [HumanMessage(content=user_message)]
            async for event in stream_subagent_events(
                self.master, messages, self.recursion_limit, on_ppt_created
            ):
                if event["event"] == TEXT_DELTA:
                    response_parts.append(event["data"].get("content", ""))
                yield event
        except GraphRecursionError:
            logger.error("[SubAgents/流式] 超出最大推理步数")
            yield make_event(ERROR, {"message": "超出最大推理步数限制"})
        except Exception as e:
            logger.error("[SubAgents/流式] 执行异常: %s", e)
            yield make_event(ERROR, {"message": f"执行异常: {e}"})
        finally:
            response_text = "".join(response_parts)
            if response_text:
                try:
                    await self.memory.save(session_id, user_message, response_text)
                except Exception as e:
                    logger.error("[SubAgents/流式] 保存记忆失败: %s", e)
            logger.info(
                "[SubAgents/流式] 会话 %s 完成, 耗时 %.2fs", session_id, time.time() - start
            )
            yield make_event(
                DONE,
                {
                    "session_id": session_id,
                    "elapsed_seconds": round(time.time() - start, 2),
                },
            )

    async def clear_history(self, session_id: str):
        await self.memory.clear(session_id)
        logger.info("[SubAgents] 会话 %s 历史已清除", session_id)
