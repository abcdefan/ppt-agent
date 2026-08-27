"""Workflow 模式的统一运行入口。

Subagents 模式的入口是 ``MasterAgent``；Workflow 模式的核心
可执行对象是编译后的 Graph。``WorkflowRunner`` 不是又一个
Agent，而是在 Graph 外层统一处理初始 State、会话记忆和流式事件。
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from app.agents.common.llm import build_llm, build_summary_llm
from app.agents.router import ChatResponder, IntentRouterService
from app.context import SessionState, SessionStateStore, SummaryBufferMemory
from app.agents.workflow.graph import build_workflow_graph
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.streaming import stream_workflow_events
from app.core.config import settings
from app.schemas.events import (
    AGENT_THINKING,
    DONE,
    ERROR,
    TEXT_DELTA,
    make_event,
)

logger = logging.getLogger(__name__)

OVERVIEW_PROMPT = (
    "你正在与 PPTCreator 多智能体团队协作。"
    "Intent Router 负责业务分流；PPT Workflow 子图中的大纲、调研、内容、"
    "配图、图表、统一编辑和美化节点负责具体工作。"
)


class WorkflowRunner:
    """Supervisor Workflow 的非流式/流式执行入口。"""

    def __init__(self):
        # settings.agent_mode=workflow 时，main.py 会在 FastAPI lifespan
        # 阶段创建本 Runner。构建 Graph 时，Supervisor Node 和全部
        # Specialist Nodes 也会一次性创建；后续请求复用该 Graph。
        self.llm = build_llm()
        self.memory = SummaryBufferMemory(summary_llm=build_summary_llm())
        self.state_store = SessionStateStore()
        self.intent_router = IntentRouterService(self.llm)
        self.chat_responder = ChatResponder(self.llm)
        self.graph = build_workflow_graph(
            self.llm,
            intent_router=self.intent_router,
            chat_responder=self.chat_responder,
        )
        self.recursion_limit = settings.agent_multi_recursion_limit
        logger.info(
            "WorkflowRunner 初始化完成: recursion_limit=%s",
            self.recursion_limit,
        )

    async def initialize(self) -> None:
        await self.intent_router.initialize()

    async def run(
        self,
        user_message: str,
        session_id: str | None = None,
        style: str | None = None,
        requested_action: str | None = None,
    ) -> str:
        """非流式执行一次完整 Workflow。"""
        session_id = session_id or "default"
        history = await self.memory.load(session_id)
        session_state = await self.state_store.load(session_id)
        initial_state = self._build_initial_state(
            user_message=user_message,
            session_id=session_id,
            style=style or session_state.style,
            history=history,
            session_state=session_state,
            requested_action=requested_action,
        )

        try:
            # Graph.ainvoke() 接收的是整个初始 State，不是直接把
            # messages 发给某个 LLM。Graph 会按边依次把 State 交给 Node。
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": self.recursion_limit},
            )
        except GraphRecursionError:
            logger.exception("[WorkflowRunner] 会话 %s 超出最大递归步数", session_id)
            return "抱歉，多智能体任务已达到最大执行步数，请简化需求后重试。"

        reply = self._last_ai_text(final_state)
        await self.memory.save(session_id, user_message, reply)
        await self.state_store.patch(
            session_id,
            active_ppt_filename=final_state.get("filename"),
            outline=final_state.get("outline"),
            style=final_state.get("style") or "business",
        )
        return reply

    async def _stream(
        self,
        user_message: str,
        session_id: str | None,
        style: str | None,
        requested_action: str | None,
        on_ppt_created: Callable[[str], Awaitable[None]] | None,
    ) -> AsyncIterator[dict]:
        """两个公开流式方法共用的内部实现。"""
        session_id = session_id or "default"
        started_at = time.monotonic()
        response_parts: list[str] = []

        yield make_event(
            AGENT_THINKING,
            {"message": "Workflow 多智能体团队开始分析任务..."},
        )

        try:
            history = await self.memory.load(session_id)
            session_state = await self.state_store.load(session_id)
            resolved_style = style or session_state.style
            await self.state_store.patch(session_id, style=resolved_style)
            initial_state = self._build_initial_state(
                user_message=user_message,
                session_id=session_id,
                style=resolved_style,
                history=history,
                session_state=session_state,
                requested_action=requested_action,
            )

            async def persist_state_patch(patch: dict) -> None:
                await self.state_store.patch(session_id, **patch)

            # streaming.py 负责把 Graph 底层事件转成项目业务事件；
            # Runner 负责继续转发，并收集文字以保存会话记忆。
            async for event in stream_workflow_events(
                graph=self.graph,
                initial_state=initial_state,
                recursion_limit=self.recursion_limit,
                on_ppt_created=on_ppt_created,
                on_state_updated=persist_state_patch,
            ):
                if event["event"] == TEXT_DELTA:
                    response_parts.append(event["data"].get("content", ""))
                yield event

        except GraphRecursionError:
            logger.exception(
                "[WorkflowRunner/流式] 会话 %s 超出最大递归步数",
                session_id,
            )
            yield make_event(
                ERROR,
                {"message": "多智能体任务已达到最大执行步数，请简化需求后重试。"},
            )
        except Exception as exc:
            logger.exception(
                "[WorkflowRunner/流式] 会话 %s 执行异常: %s",
                session_id,
                exc,
            )
            yield make_event(ERROR, {"message": f"执行异常: {exc}"})
        finally:
            response_text = "".join(response_parts).strip()
            if response_text:
                try:
                    await self.memory.save(session_id, user_message, response_text)
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/流式] 会话 %s 保存记忆失败",
                        session_id,
                    )

            elapsed_seconds = round(time.monotonic() - started_at, 2)
            logger.info(
                "[WorkflowRunner/流式] 会话 %s 完成, 耗时 %.2fs",
                session_id,
                elapsed_seconds,
            )

        # 不在 finally 中 yield，避免浏览器断开后关闭 async generator 时
        # 又尝试产生新数据。正常结束或已转成 ERROR 时才发 DONE。
        yield make_event(
            DONE,
            {
                "session_id": session_id,
                "elapsed_seconds": elapsed_seconds,
            },
        )

    async def run_stream(
        self,
        user_message: str,
        session_id: str | None = None,
        style: str | None = None,
        requested_action: str | None = None,
    ) -> AsyncIterator[dict]:
        """普通流式执行入口。"""
        async for event in self._stream(
            user_message=user_message,
            session_id=session_id,
            style=style,
            requested_action=requested_action,
            on_ppt_created=None,
        ):
            yield event

    async def clear_history(self, session_id: str) -> None:
        """清除指定会话记忆。"""
        await self.memory.clear(session_id)

    @staticmethod
    def _build_initial_state(
        user_message: str,
        session_id: str,
        style: str,
        history: list,
        session_state: SessionState,
        requested_action: str | None,
    ) -> WorkflowState:
        """把用户请求和跨请求历史组装成 Graph 的初始 State。"""
        return {
            "messages": [
                SystemMessage(content=OVERVIEW_PROMPT),
                *history,
                HumanMessage(content=user_message),
            ],
            "user_message": user_message,
            "session_id": session_id,
            "style": style,
            "outline": session_state.outline,
            "research_report": None,
            "filename": session_state.active_ppt_filename,
            "asset_operations": [],
            "asset_apply_status": "pending",
            "applied_operation_ids": [],
            "completed_agents": [],
            "next": None,
            "requested_action": requested_action,
            "intent": None,
            "route_source": None,
            "route_confidence": None,
            "route_reason": None,
        }

    @staticmethod
    def _last_ai_text(final_state: WorkflowState) -> str:
        """从最终 State 取最后一条 Specialist AI 文本。"""
        for message in reversed(final_state.get("messages", [])):
            if isinstance(message, AIMessage) and not message.tool_calls:
                text = str(message.text).strip()
                if text:
                    return text
        return "任务执行完成，但没有生成文字回复。"
