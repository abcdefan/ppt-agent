"""Workflow 模式的统一运行入口。

Subagents 模式的入口是 ``MasterAgent``；Workflow 模式的核心
可执行对象是编译后的 Graph。``WorkflowRunner`` 不是又一个
Agent，而是在 Graph 外层统一处理初始 State、会话记忆和流式事件。
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from app.agents.common.llm import build_llm, build_summary_llm
from app.agents.router import ChatResponder, IntentRouterService
from app.agents.workflow.graph import build_workflow_graph
from app.agents.workflow.state import WorkflowState
from app.agents.workflow.streaming import stream_workflow_events
from app.agents.workflow.subgraphs import (
    build_debug_ppt_creation_subgraph,
    build_ppt_creation_subgraph,
)
from app.context import (
    PptRecordStore,
    SessionState,
    SessionStateStore,
    SummaryBufferMemory,
)
from app.core.config import settings
from app.core.database import database
from app.repositories import ChatMessageRepository
from app.schemas.events import (
    AGENT_THINKING,
    DONE,
    ERROR,
    INPUT_REQUIRED,
    INTENT_ROUTED,
    TEXT_DELTA,
    make_event,
)
from app.services import (
    PptContextService,
    PptOwnershipError,
    WorkflowRunConflictError,
)

logger = logging.getLogger(__name__)

class WorkflowRunner:
    """Workflow Graph 的非流式/流式执行入口。"""

    def __init__(self, *, checkpointer: BaseCheckpointSaver):
        # settings.agent_mode=workflow 时，main.py 会在 FastAPI lifespan
        # 阶段创建本 Runner。构建 Graph 时，Planner、Edit Supervisor 和全部
        # Specialist Nodes 也会一次性创建；后续请求复用该 Graph。
        self.llm = build_llm()
        # Edit 目标核验是分类任务：使用低温、非流式模型，避免判断随机波动。
        self.target_match_llm = build_llm(temperature=0.0, streaming=False)
        self.memory = SummaryBufferMemory(summary_llm=build_summary_llm())
        self.state_store = SessionStateStore()
        self.ppt_record_store = PptRecordStore()
        self.ppt_context_service = PptContextService(database)
        self.chat_message_repository = ChatMessageRepository(database)
        self.intent_router = IntentRouterService(self.llm)
        self.chat_responder = ChatResponder(self.llm)
        self.checkpointer = checkpointer
        if settings.agent_ppt_subgraph_mode == "debug":
            self.create_subgraph = build_debug_ppt_creation_subgraph(
                self.llm,
                self.ppt_context_service,
            )
        else:
            self.create_subgraph = build_ppt_creation_subgraph(
                self.llm,
                self.ppt_context_service,
            )
        self.graph = build_workflow_graph(
            self.llm,
            intent_router=self.intent_router,
            chat_responder=self.chat_responder,
            session_store=self.state_store,
            ppt_record_store=self.ppt_record_store,
            ppt_context_service=self.ppt_context_service,
            creation_subgraph=self.create_subgraph,
            target_match_llm=self.target_match_llm,
            checkpointer=self.checkpointer,
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
        user_id: int,
        session_id: str | None = None,
        style: str | None = None,
        requested_action: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """非流式执行一次完整 Workflow。"""
        session_id = session_id or "default"
        effective_run_id = run_id or f"run-{uuid4().hex}"
        await self.ppt_context_service.ensure_session(
            session_id=session_id,
            user_id=user_id,
        )
        user_record = await self.chat_message_repository.create(
            session_id=session_id,
            message_role="USER",
            content=user_message,
            metadata={
                "requested_action": requested_action,
            },
        )
        conversation_history = await self.memory.load(session_id)
        session_state = await self.state_store.load(session_id)
        initial_state = self._build_initial_state(
            user_message=user_message,
            session_id=session_id,
            style=style or "business",
            conversation_history=conversation_history,
            session_state=session_state,
            requested_action=requested_action,
            user_id=user_id,
            run_id=effective_run_id,
        )

        try:
            # Graph.ainvoke() 接收整个初始 State。每个 Node 从同一份
            # conversation_history 快照和结构化业务字段中选择所需上下文。
            final_state = await self.graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": self.recursion_limit,
                    "configurable": {"thread_id": effective_run_id},
                },
                durability="exit",
            )
        except GraphRecursionError:
            logger.exception("[WorkflowRunner] 会话 %s 超出最大递归步数", session_id)
            await self._mark_run_failed_safely(
                run_id=effective_run_id,
                user_id=user_id,
                error_code="GRAPH_RECURSION_LIMIT",
                error_message="Workflow 超出最大递归步数",
            )
            reply = "抱歉，多智能体任务已达到最大执行步数，请简化需求后重试。"
            await self._archive_assistant_message(
                session_id=session_id,
                user_id=user_id,
                user_message_id=user_record["id"],
                run_id=effective_run_id,
                content=reply,
                message_type="ERROR",
                message_status="FAILED",
            )
            return reply
        except Exception:
            logger.exception("[WorkflowRunner] 会话 %s 执行异常", session_id)
            await self._mark_run_failed_safely(
                run_id=effective_run_id,
                user_id=user_id,
                error_code="WORKFLOW_EXCEPTION",
                error_message="Workflow 执行异常",
            )
            await self._archive_assistant_message(
                session_id=session_id,
                user_id=user_id,
                user_message_id=user_record["id"],
                run_id=effective_run_id,
                content="Workflow 执行异常",
                message_type="ERROR",
                message_status="FAILED",
            )
            raise

        reply = self._final_response_text(final_state)
        is_execution = (
            final_state.get("intent") in {"create", "edit"}
            and final_state.get("execute") is True
        )
        await self._archive_assistant_message(
            session_id=session_id,
            user_id=user_id,
            user_message_id=user_record["id"],
            run_id=effective_run_id,
            content=reply,
            message_type="WORKFLOW_RESULT" if is_execution else "TEXT",
            metadata={
                "intent": final_state.get("intent"),
                "execute": final_state.get("execute", False),
                "ppt_id": final_state.get("ppt_id"),
            },
        )
        await self.memory.save(session_id, user_message, reply)
        return reply

    async def _stream(
        self,
        user_message: str,
        session_id: str | None,
        style: str | None,
        requested_action: str | None,
        on_ppt_created: Callable[[str], Awaitable[None]] | None,
        user_id: int,
        run_id: str,
    ) -> AsyncIterator[dict]:
        """两个公开流式方法共用的内部实现。"""
        session_id = session_id or "default"
        started_at = time.monotonic()
        response_parts: list[str] = []
        routed_intent: str | None = None
        routed_execute = False
        user_message_id: int | None = None
        archived_error: str | None = None
        stream_completed = False
        waiting_input = False

        yield make_event(
            AGENT_THINKING,
            {"message": "Workflow 多智能体团队开始分析任务..."},
        )

        try:
            await self.ppt_context_service.ensure_session(
                session_id=session_id,
                user_id=user_id,
            )
            user_record = await self.chat_message_repository.create(
                session_id=session_id,
                message_role="USER",
                content=user_message,
                metadata={
                    "requested_action": requested_action,
                },
            )
            user_message_id = int(user_record["id"])
            conversation_history = await self.memory.load(session_id)
            session_state = await self.state_store.load(session_id)
            resolved_style = style or "business"
            initial_state = self._build_initial_state(
                user_message=user_message,
                session_id=session_id,
                style=resolved_style,
                conversation_history=conversation_history,
                session_state=session_state,
                requested_action=requested_action,
                user_id=user_id,
                run_id=run_id,
            )

            # streaming.py 负责把 Graph 底层事件转成项目业务事件；
            # Runner 负责继续转发，并收集文字以保存会话记忆。
            async for event in stream_workflow_events(
                graph=self.graph,
                initial_state=initial_state,
                recursion_limit=self.recursion_limit,
                thread_id=run_id,
                on_ppt_created=on_ppt_created,
            ):
                if event["event"] == TEXT_DELTA:
                    response_parts.append(event["data"].get("content", ""))
                elif event["event"] == INTENT_ROUTED:
                    routed_intent = event["data"].get("intent")
                    routed_execute = event["data"].get("execute") is True
                elif event["event"] == INPUT_REQUIRED:
                    waiting_input = True
                yield event
            stream_completed = True

        except GraphRecursionError:
            logger.exception(
                "[WorkflowRunner/流式] 会话 %s 超出最大递归步数",
                session_id,
            )
            yield make_event(
                ERROR,
                {"message": "多智能体任务已达到最大执行步数，请简化需求后重试。"},
            )
            archived_error = "多智能体任务已达到最大执行步数，请简化需求后重试。"
            await self._mark_run_failed_safely(
                run_id=run_id,
                user_id=user_id,
                error_code="GRAPH_RECURSION_LIMIT",
                error_message="Workflow 超出最大递归步数",
            )
        except Exception as exc:
            logger.exception(
                "[WorkflowRunner/流式] 会话 %s 执行异常",
                session_id,
            )
            await self._mark_run_failed_safely(
                run_id=run_id,
                user_id=user_id,
                error_code="WORKFLOW_EXCEPTION",
                error_message=str(exc),
            )
            archived_error = f"执行异常: {exc}"
            yield make_event(ERROR, {"message": archived_error})
        finally:
            response_text = "".join(response_parts).strip()
            archive_content = archived_error or (
                response_text if stream_completed else ""
            )
            if archive_content:
                try:
                    await self._archive_assistant_message(
                        session_id=session_id,
                        user_id=user_id,
                        user_message_id=user_message_id,
                        run_id=run_id,
                        content=archive_content,
                        message_type=(
                            "ERROR"
                            if archived_error
                            else (
                                "WORKFLOW_RESULT"
                                if (
                                    routed_intent in {"create", "edit"}
                                    and routed_execute
                                )
                                else "TEXT"
                            )
                        ),
                        message_status="FAILED" if archived_error else "COMPLETED",
                        metadata={
                            "intent": routed_intent,
                            "execute": routed_execute,
                        },
                    )
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/流式] 会话 %s 归档消息失败",
                        session_id,
                    )
            if response_text and stream_completed and not archived_error:
                try:
                    await self.memory.save(session_id, user_message, response_text)
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/流式] 会话 %s 保存记忆失败",
                        session_id,
                    )
            elif waiting_input and user_message_id is not None:
                # 暂停时还没有最终 Assistant 回复，但 Edit Run 已创建。
                # 先绑定原始 USER 消息，保证 resume 后的结果与同一 Run
                # 在 MySQL 中可以完整关联。
                try:
                    await self.chat_message_repository.attach_run(
                        message_id=user_message_id,
                        run_id=run_id,
                    )
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/流式] 会话 %s 绑定等待 Run 失败",
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
                "run_id": (
                    run_id
                    if routed_intent in {"create", "edit"} and routed_execute
                    else None
                ),
                "elapsed_seconds": elapsed_seconds,
                "status": "WAITING_INPUT" if waiting_input else "COMPLETED",
            },
        )

    async def run_stream(
        self,
        user_message: str,
        user_id: int,
        session_id: str | None = None,
        style: str | None = None,
        requested_action: str | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """普通流式执行入口。"""
        async for event in self._stream(
            user_message=user_message,
            session_id=session_id,
            style=style,
            requested_action=requested_action,
            on_ppt_created=None,
            user_id=user_id,
            run_id=run_id or f"run-{uuid4().hex}",
        ):
            yield event

    async def run_edit_resume_stream(
        self,
        *,
        user_id: int,
        run_id: str,
        ppt_id: str,
        revision: int,
    ) -> AsyncIterator[dict]:
        """Submit a PPT selection and continue the interrupted parent graph."""
        started_at = time.monotonic()
        session_id: str | None = None
        checkpoint_thread_id: str | None = None
        original_message = ""
        execution_started = False
        archive_content: str | None = None
        archive_failed = False
        response_parts: list[str] = []

        yield make_event(
            AGENT_THINKING,
            {"message": "已收到 PPT 选择，正在恢复编辑流程..."},
        )

        try:
            run = await self.ppt_context_service.load_edit_resume_context(
                user_id=user_id,
                run_id=run_id,
                expected_revision=revision,
            )
            session_id = str(run["session_id"])
            checkpoint_thread_id = str(run["checkpoint_thread_id"])
            input_payload = run.get("input_payload_json") or {}
            if isinstance(input_payload, dict):
                original_message = str(input_payload.get("message") or "")
            execution_started = True

            async for event in stream_workflow_events(
                graph=self.graph,
                initial_state=Command(
                    resume={"ppt_id": ppt_id, "revision": revision}
                ),
                recursion_limit=self.recursion_limit,
                thread_id=checkpoint_thread_id,
            ):
                if event["event"] == TEXT_DELTA:
                    response_parts.append(event["data"].get("content", ""))
                yield event

            refreshed_run = await self.ppt_context_service.run_repository.get(
                run_id=run_id,
                user_id=user_id,
            )
            if refreshed_run and refreshed_run.get("run_status") == "SUCCEEDED":
                archive_content = "".join(response_parts).strip()
                if not archive_content:
                    archive_content = "PPT 已修改完成。"
                    yield make_event(TEXT_DELTA, {"content": archive_content})
            else:
                archive_failed = True
                archive_content = str(
                    (refreshed_run or {}).get("error_message")
                    or "PPT 编辑流程未成功完成"
                )
                yield make_event(
                    ERROR,
                    {"message": archive_content, "run_id": run_id},
                )

        except (WorkflowRunConflictError, PptOwnershipError) as exc:
            logger.info("[WorkflowRunner/Edit 恢复] Run %s 不可恢复: %s", run_id, exc)
            archive_failed = True
            archive_content = str(exc)
            yield make_event(
                ERROR,
                {"message": archive_content, "run_id": run_id},
            )
        except GraphRecursionError:
            logger.exception(
                "[WorkflowRunner/Edit 恢复] Run %s 超出最大递归步数",
                run_id,
            )
            archive_failed = True
            archive_content = "PPT 编辑恢复达到最大执行步数"
            if execution_started:
                await self._mark_run_failed_safely(
                    run_id=run_id,
                    user_id=user_id,
                    error_code="GRAPH_RECURSION_LIMIT",
                    error_message=archive_content,
                )
            yield make_event(
                ERROR,
                {"message": archive_content, "run_id": run_id},
            )
        except Exception as exc:
            logger.exception("[WorkflowRunner/Edit 恢复] Run %s 执行异常", run_id)
            archive_failed = True
            archive_content = f"PPT 编辑恢复失败：{exc}"
            if execution_started:
                await self._mark_run_failed_safely(
                    run_id=run_id,
                    user_id=user_id,
                    error_code="WORKFLOW_RESUME_EXCEPTION",
                    error_message=str(exc),
                )
            yield make_event(
                ERROR,
                {"message": archive_content, "run_id": run_id},
            )
        finally:
            if session_id and archive_content:
                try:
                    await self._archive_assistant_message(
                        session_id=session_id,
                        user_id=user_id,
                        user_message_id=None,
                        run_id=run_id,
                        content=archive_content,
                        message_type="ERROR" if archive_failed else "WORKFLOW_RESULT",
                        message_status="FAILED" if archive_failed else "COMPLETED",
                        metadata={
                            "intent": "edit",
                            "resume": True,
                            "ppt_id": ppt_id,
                        },
                    )
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/Edit 恢复] Run %s 归档消息失败",
                        run_id,
                    )
            if (
                session_id
                and original_message
                and archive_content
                and not archive_failed
            ):
                try:
                    await self.memory.save(
                        session_id,
                        original_message,
                        archive_content,
                    )
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/Edit 恢复] Run %s 保存会话记忆失败",
                        run_id,
                    )

        yield make_event(
            DONE,
            {
                "session_id": session_id,
                "run_id": run_id,
                "ppt_id": ppt_id,
                "elapsed_seconds": round(time.monotonic() - started_at, 2),
                "status": "FAILED" if archive_failed else "COMPLETED",
            },
        )

    async def run_create_resume_stream(
        self,
        *,
        user_id: int,
        run_id: str,
    ) -> AsyncIterator[dict]:
        """加载应用层断点，并直接从同一个 Create Subgraph 继续流式执行。"""
        started_at = time.monotonic()
        session_id: str | None = None
        ppt_id: str | None = None
        execution_started = False
        archive_content: str | None = None
        archive_failed = False

        yield make_event(
            AGENT_THINKING,
            {"message": "正在加载上次进度并继续生成 PPT..."},
        )

        try:
            resume_state = await self.load_create_resume_state(
                user_id=user_id,
                run_id=run_id,
            )
            session_id = resume_state["session_id"]
            ppt_id = resume_state["ppt_id"]
            execution_started = True

            async for event in stream_workflow_events(
                graph=self.create_subgraph,
                initial_state=resume_state,
                recursion_limit=self.recursion_limit,
            ):
                yield event

            refreshed_run = await self.ppt_context_service.run_repository.get(
                run_id=run_id,
                user_id=user_id,
            )
            if refreshed_run and refreshed_run.get("run_status") == "SUCCEEDED":
                refreshed_ppt = await self.ppt_context_service.ppt_repository.get(
                    ppt_id=ppt_id,
                    user_id=user_id,
                )
                filename = (
                    refreshed_ppt.get("current_filename") if refreshed_ppt else None
                )
                archive_content = (
                    f"PPT 已从断点继续生成完成：{filename}"
                    if filename
                    else "PPT 已从断点继续生成完成。"
                )
                yield make_event(TEXT_DELTA, {"content": archive_content})
            else:
                archive_failed = True
                archive_content = str(
                    (refreshed_run or {}).get("error_message")
                    or "PPT 断点重跑未成功完成"
                )
                yield make_event(ERROR, {"message": archive_content})

        except (WorkflowRunConflictError, PptOwnershipError) as exc:
            logger.info("[WorkflowRunner/恢复] Run %s 不可恢复: %s", run_id, exc)
            archive_failed = True
            archive_content = str(exc)
            yield make_event(ERROR, {"message": archive_content})
        except GraphRecursionError:
            logger.exception("[WorkflowRunner/恢复] Run %s 超出最大递归步数", run_id)
            archive_failed = True
            archive_content = "PPT 断点重跑达到最大执行步数"
            if execution_started:
                await self._mark_run_failed_safely(
                    run_id=run_id,
                    user_id=user_id,
                    error_code="GRAPH_RECURSION_LIMIT",
                    error_message=archive_content,
                )
            yield make_event(ERROR, {"message": archive_content})
        except Exception as exc:
            logger.exception("[WorkflowRunner/恢复] Run %s 执行异常", run_id)
            archive_failed = True
            archive_content = f"PPT 断点重跑失败：{exc}"
            # 加载/校验失败时还没有开始执行，不能误伤数据库中的其他 Run。
            if execution_started:
                await self._mark_run_failed_safely(
                    run_id=run_id,
                    user_id=user_id,
                    error_code="WORKFLOW_RESUME_EXCEPTION",
                    error_message=str(exc),
                )
            yield make_event(ERROR, {"message": archive_content})
        finally:
            if session_id and archive_content:
                try:
                    await self._archive_assistant_message(
                        session_id=session_id,
                        user_id=user_id,
                        user_message_id=None,
                        run_id=run_id,
                        content=archive_content,
                        message_type="ERROR" if archive_failed else "WORKFLOW_RESULT",
                        message_status="FAILED" if archive_failed else "COMPLETED",
                        metadata={"intent": "create", "resume": True, "ppt_id": ppt_id},
                    )
                except Exception:
                    logger.exception(
                        "[WorkflowRunner/恢复] Run %s 归档消息失败",
                        run_id,
                    )

        yield make_event(
            DONE,
            {
                "session_id": session_id,
                "run_id": run_id,
                "ppt_id": ppt_id,
                "elapsed_seconds": round(time.monotonic() - started_at, 2),
            },
        )

    async def clear_history(self, session_id: str) -> None:
        """清除指定会话记忆。"""
        await self.memory.clear(session_id)

    async def load_create_resume_state(
        self,
        *,
        user_id: int,
        run_id: str,
    ) -> WorkflowState:
        """从 MySQL 加载 Create Run/PPT，并组装可直接传给子图的 State。"""
        context = await self.ppt_context_service.load_create_resume_context(
            user_id=user_id,
            run_id=run_id,
        )
        run = context["run"]
        session_id = str(run["session_id"])
        conversation_history = await self.memory.load(session_id)
        return self._build_create_resume_state(
            user_id=user_id,
            run=context["run"],
            ppt=context["ppt"],
            conversation_history=conversation_history,
        )

    @staticmethod
    def _build_initial_state(
        user_message: str,
        session_id: str,
        style: str,
        conversation_history: list[BaseMessage],
        session_state: SessionState,
        requested_action: str | None,
        user_id: int,
        run_id: str,
    ) -> WorkflowState:
        """在进入 Graph 前固定本轮会话快照，并初始化全部业务字段。"""
        return {
            # Redis history 只在本轮入口读取一次，不含当前 user_message。
            "conversation_history": list(conversation_history),
            # 当前请求单独保存，节点构建 Prompt 时再与 history 组合。
            "user_message": user_message,
            # 只有统一 Reply Node 负责写最终用户可见结果。
            "final_response": None,
            "final_response_mode": None,
            "user_id": user_id,
            "run_id": run_id,
            "session_id": session_id,
            "style": style,
            "active_ppt_id": session_state.active_ppt_id,
            "ppt_id": None,
            "ppt_context_error": None,
            "workflow_error": None,
            "edit_target_matches_active": None,
            "edit_target_match_reason": None,
            "research_report": None,
            "outline": None,
            "filename": None,
            "slides_manifest": None,
            "asset_operations": [],
            "asset_apply_status": "pending",
            "applied_operation_ids": [],
            "asset_tasks": [],
            "completed_agents": [],
            "required_stages": [],
            "completed_stages": [],
            "requirements_initialized": False,
            "create_finalized": False,
            "attempt_error": None,
            "attempt_counts": {},
            "edit_next": None,
            "requested_action": requested_action,
            "intent": None,
            "execute": False,
            "route_source": None,
            "route_confidence": None,
            "route_reason": None,
        }

    @staticmethod
    def _build_create_resume_state(
        *,
        user_id: int,
        run: dict,
        ppt: dict,
        conversation_history: list[BaseMessage],
    ) -> WorkflowState:
        """把 workflow_run 的进度和 ppt_record 的产物还原为 Create State。"""
        ppt_id = str(run["ppt_id"])
        run_id = str(run["run_id"])
        session_id = str(run["session_id"])
        input_payload = run.get("input_payload_json") or {}
        user_message = str(
            input_payload.get("message")
            or ppt.get("title")
            or "继续生成 PPT"
        )
        style = str(ppt.get("style") or input_payload.get("style") or "business")
        completed_stages = list(run.get("completed_stages_json") or [])
        required_stages = list(run.get("required_stages_json") or [])
        completed = set(completed_stages)
        current_stage = str(run.get("current_stage") or "").upper()

        state = WorkflowRunner._build_initial_state(
            user_message=user_message,
            session_id=session_id,
            style=style,
            conversation_history=conversation_history,
            session_state=SessionState(
                active_ppt_id=ppt_id,
                ppt_ids=[ppt_id],
            ),
            requested_action="create",
            user_id=user_id,
            run_id=run_id,
        )
        state.update(
            {
                "active_ppt_id": ppt_id,
                "ppt_id": ppt_id,
                "research_report": ppt.get("research_report_json"),
                "outline": ppt.get("outline_json"),
                "filename": ppt.get("current_filename"),
                "slides_manifest": ppt.get("slides_manifest_json"),
                "asset_apply_status": (
                    "succeeded" if "assets" in completed else "pending"
                ),
                "completed_agents": [
                    stage
                    for stage in completed_stages
                    if stage in {"research", "outline", "content", "beautify"}
                ],
                "required_stages": required_stages,
                "completed_stages": completed_stages,
                # Planner 本身没有里程碑。只有可选阶段已经落库，或者最终
                # Persist 已开始时，才能确认其规划结果已经持久化。
                "requirements_initialized": bool(
                    {"assets", "beautify"} & completed
                    or current_stage == "FINALIZE"
                ),
                "intent": "create",
                "execute": True,
                "route_source": "explicit",
                "route_confidence": 1.0,
                "route_reason": "从 MySQL 恢复 Create Workflow",
            }
        )
        return state

    async def _mark_run_failed_safely(
        self,
        *,
        run_id: str,
        user_id: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """Graph 在最终 persist 前崩溃时，尽力关闭已创建的 Run。"""
        try:
            run = await self.ppt_context_service.run_repository.get(
                run_id=run_id,
                user_id=user_id,
            )
            if run is None or run.get("run_status") not in {"CREATED", "RUNNING"}:
                return
            await self.ppt_context_service.fail_run(
                run_id=run_id,
                user_id=user_id,
                error_code=error_code,
                error_message=error_message[:2000],
            )
        except WorkflowRunConflictError:
            logger.info("Workflow Run 已由其他执行推进，跳过失败回写: %s", run_id)
        except Exception:
            logger.exception("Workflow Run 失败状态回写异常: %s", run_id)

    async def _archive_assistant_message(
        self,
        *,
        session_id: str,
        user_id: int,
        user_message_id: int | None,
        run_id: str,
        content: str,
        message_type: str,
        message_status: str = "COMPLETED",
        metadata: dict | None = None,
    ) -> None:
        """归档最终回复；只有 Run 已实际创建时才写入外键。"""
        linked_run_id: str | None = None
        run = await self.ppt_context_service.run_repository.get(
            run_id=run_id,
            user_id=user_id,
        )
        if run is not None:
            linked_run_id = run_id
            if user_message_id is not None:
                await self.chat_message_repository.attach_run(
                    message_id=user_message_id,
                    run_id=run_id,
                )
        await self.chat_message_repository.create(
            session_id=session_id,
            run_id=linked_run_id,
            message_role="ASSISTANT",
            message_type=message_type,
            content=content,
            metadata=metadata,
            message_status=message_status,
        )

    @staticmethod
    def _final_response_text(final_state: WorkflowState) -> str:
        """从 Reply Node 的显式输出字段取得最终用户回复。"""
        response = final_state.get("final_response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        return "任务执行完成，但没有生成文字回复。"
