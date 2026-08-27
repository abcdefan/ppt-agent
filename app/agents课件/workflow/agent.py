"""workflow 模式入口 — Custom workflow（StateGraph + Supervisor 路由）。

对外方法签名与 SubAgents / 旧 PPTAgent / LangChainAgent 完全一致：
  - run(user_message, session_id) -> str
  - run_stream(user_message, session_id) -> AsyncIterator[dict]
  - run_ppt_stream(user_message, session_id, on_ppt_created) -> AsyncIterator[dict]
  - clear_history(session_id)
"""

import logging
import time
from typing import AsyncIterator, Awaitable, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import settings
from app.schemas.events import AGENT_THINKING, DONE, ERROR, TEXT_DELTA, make_event
from app.utils.memory import SummaryBufferMemory

from app.agents.common.llm import build_llm, build_summary_llm
from app.agents.workflow.graph import build_graph
from app.agents.workflow.streaming import stream_events_to_sse

# 导入 GraphRecursionError（langgraph 1.x）
try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # 兜底：旧版本路径
    from langgraph.graph.exc import GraphRecursionError  # type: ignore

logger = logging.getLogger(__name__)

# 多智能体团队总览提示（作为消息流开头的 SystemMessage）
_OVERVIEW_PROMPT = (
    "你是 PPTCreator 多智能体团队（协调员 + 内容/配图/图表/美化专家）。"
    "请通过协作完成用户需求，用简洁专业的中文汇报。"
)


class MultiAgent:
    """workflow 模式：Supervisor + 4 Specialist（Custom workflow）。"""

    def __init__(self):
        self.llm = build_llm()
        self.memory = SummaryBufferMemory(summary_llm=build_summary_llm())
        self.graph = build_graph(self.llm)
        self.recursion_limit = settings.agent_multi_recursion_limit
        logger.info(
            "MultiAgent(workflow) 初始化完成 (supervisor + outline/content/image/chart/beautify), "
            "recursion_limit=%d",
            self.recursion_limit,
        )

    # ============================================================
    # 非流式
    # ============================================================
    async def run(self, user_message: str, session_id: Optional[str] = None) -> str:
        session_id = session_id or "default"
        logger.info("[MultiAgent] 会话 %s 收到消息: %s", session_id, user_message[:100])

        history = await self.memory.load(session_id)
        state = self._init_state(user_message, session_id, history)

        try:
            final = await self.graph.ainvoke(
                state, {"recursion_limit": self.recursion_limit}
            )
        except GraphRecursionError:
            logger.error("[MultiAgent] 超出最大推理步数")
            return "抱歉，任务过于复杂，已达到最大推理步数限制，请尝试简化需求。"

        reply = self._last_ai_text(final)
        await self.memory.save(session_id, user_message, reply)
        return reply

    # ============================================================
    # 流式（对话）
    # ============================================================
    async def run_stream(
        self, user_message: str, session_id: Optional[str] = None
    ) -> AsyncIterator[dict]:
        async for event in self._stream(user_message, session_id, None):
            yield event

    # ============================================================
    # 流式（PPT 创作，透传 on_ppt_created 回调）
    # ============================================================
    async def run_ppt_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        on_ppt_created: Optional[Callable[[str], Awaitable[None]]] = None,
        style: str = "business",
    ) -> AsyncIterator[dict]:
        async for event in self._stream(user_message, session_id, on_ppt_created, style):
            yield event

    # ============================================================
    # 流式核心
    # ============================================================
    async def _stream(
        self,
        user_message: str,
        session_id: Optional[str],
        on_ppt_created: Optional[Callable[[str], Awaitable[None]]],
        style: str = "business",
    ) -> AsyncIterator[dict]:
        session_id = session_id or "default"
        start = time.time()
        response_parts: list[str] = []

        logger.info(
            "[MultiAgent/流式] 会话 %s 收到消息: %s", session_id, user_message[:100]
        )

        yield make_event(AGENT_THINKING, {"message": "多智能体团队开始协作..."})

        try:
            history = await self.memory.load(session_id)
            state = self._init_state(user_message, session_id, history, style)

            async for event in stream_events_to_sse(
                self.graph, state, self.recursion_limit, on_ppt_created
            ):
                # 收集文本片段，用于结束后写回记忆
                if event["event"] == TEXT_DELTA:
                    response_parts.append(event["data"].get("content", ""))
                yield event

        except GraphRecursionError:
            logger.error("[MultiAgent/流式] 超出最大推理步数")
            yield make_event(ERROR, {"message": "超出最大推理步数限制"})
        except Exception as e:
            logger.error("[MultiAgent/流式] 执行异常: %s", e)
            yield make_event(ERROR, {"message": f"执行异常: {e}"})
        finally:
            # 保存记忆（失败不影响已推送的事件）
            response_text = "".join(response_parts)
            if response_text:
                try:
                    await self.memory.save(session_id, user_message, response_text)
                except Exception as e:
                    logger.error("[MultiAgent/流式] 保存记忆失败: %s", e)
            logger.info(
                "[MultiAgent/流式] 会话 %s 完成, 耗时 %.2fs",
                session_id, time.time() - start,
            )
            yield make_event(
                DONE,
                {
                    "session_id": session_id,
                    "elapsed_seconds": round(time.time() - start, 2),
                },
            )

    # ============================================================
    # 清除历史
    # ============================================================
    async def clear_history(self, session_id: str):
        await self.memory.clear(session_id)
        logger.info("[MultiAgent] 会话 %s 历史已清除", session_id)

    # ============================================================
    # 辅助
    # ============================================================
    @staticmethod
    def _init_state(
        user_message: str, session_id: str, history: list, style: str = "business"
    ) -> dict:
        """构造初始 AgentState。"""
        messages = [SystemMessage(content=_OVERVIEW_PROMPT)]
        messages.extend(history)
        messages.append(HumanMessage(content=user_message))
        return {
            "messages": messages,
            "next": None,
            "intent": None,
            "outline": None,
            "research": None,
            "filename": None,
            "style": style,
            "user_message": user_message,
            "session_id": session_id,
        }

    @staticmethod
    def _last_ai_text(final_state: dict) -> str:
        """从最终 state 取最后一条有内容的 AIMessage 文本。"""
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                if msg.content.strip():
                    return msg.content
        return "任务执行完成，但未生成文字回复。"
