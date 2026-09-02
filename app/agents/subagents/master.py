"""PPT 多智能体团队的 Master Agent。"""

import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from app.agents.common.llm import build_llm, build_summary_llm
from app.agents.router import ChatResponder, IntentRouterService, RouteContext
from app.context import (
    PptRecord,
    PptRecordStore,
    SessionStateStore,
    SummaryBufferMemory,
)
from app.agents.subagents.delegation_tools import build_delegation_tools
from app.agents.subagents.streaming import stream_subagent_events
from app.core.config import settings
from app.schemas.events import (
    AGENT_THINKING,
    DONE,
    ERROR,
    INTENT_ROUTED,
    TEXT_DELTA,
    make_event,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "ppt_master"


SYSTEM_PROMPT = """
你是 PPTCreator 多智能体团队的 Master Agent，负责理解用户目标、拆分任务、
选择合适的专业子 Agent，并汇总执行结果。你不直接生成或修改 PPT 文件。

可调用的专业子 Agent：
- outline_agent_tool：规划 PPT 故事线和页面大纲。
- research_agent_tool：按需联网检索事实、最新数据、政策、趋势和案例。
- content_agent_tool：根据大纲生成内容和基础 PPTX 文件。
- image_agent_tool：为已有 PPT 添加配图。
- chart_agent_tool：为已有 PPT 添加数据图表。
- beautify_agent_tool：对已有 PPT 进行布局和视觉美化。

完整创建 PPT 时严格按照以下顺序协作：
1. 调用 research_agent_tool，根据用户原始需求完成联网调研。
2. 将用户需求和调研报告全文交给 outline_agent_tool 生成大纲。
3. 将主题、风格、页数、完整大纲和调研报告全文交给 content_agent_tool 生成 PPTX。
4. 根据用户需求按需调用 image_agent_tool 和 chart_agent_tool。
5. 在其他文件修改完成后调用 beautify_agent_tool 做最终美化。

规则：
- 调用下一个子 Agent 时，必须传递它需要的上游结果。
- research_agent_tool 需要用户原始需求；outline_agent_tool 需要完整调研报告；
  content_agent_tool 需要完整大纲和已经执行的调研报告；image_agent_tool、chart_agent_tool 和
  beautify_agent_tool 需要真实 PPT 文件名。
- research 返回 unavailable 时仍继续 outline 和 content，但必须告诉它们不得猜测
  外部数据或来源；同一轮不要重复调用 research。
- 不要编造大纲、工具结果或文件名。
- 子 Agent 返回失败时，应说明失败原因，不要假装任务已经完成。
- 用户只需要部分能力时，只调用相关子 Agent，不必执行完整流程。
- 最终用简洁、专业的中文向用户汇报执行结果和文件信息。
"""


def _extract_final_reply(result: dict, input_message_count: int) -> str:
    """从本次 Agent 新生成的消息中提取最终 AI 文字。"""
    result_messages = result.get("messages", [])

    # result 中同时包含传入的历史消息和本次新生成的消息。只检查新增部分，
    # 可以避免本轮没有生成文字时误拿历史 AIMessage 作为本轮结果。
    generated_messages = result_messages[input_message_count:]
    for message in reversed(generated_messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            reply = str(message.text).strip()
            if reply:
                return reply

    return "任务执行完成，但 Master Agent 没有生成文字回复。"


def _extract_ppt_filename(result: dict) -> str | None:
    """非流式模式从 Master 可见的工具结果中尽力提取 PPTX 文件名。"""
    for message in reversed(result.get("messages", [])):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        matches = re.findall(r"[^\s/\\:，。]+\.pptx", content, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
    return None


def _build_business_context(
    record: PptRecord,
    intent: str,
) -> SystemMessage:
    return SystemMessage(
        content=(
            "[当前会话业务状态]\n"
            f"本轮入口意图：{intent}\n"
            f"目标 PPT ID：{record.ppt_id}\n"
            f"目标 PPT 文件：{record.filename or '尚未生成'}\n"
            f"默认风格：{record.style}\n"
            "本轮 intent=create 表示必须新建 PPT，不要把此前活动文件当作本轮产物。"
        )
    )


class MasterAgent:
    """Subagents-as-Tools 模式的统一执行入口。"""

    def __init__(self):
        self.llm = build_llm()
        self.memory = SummaryBufferMemory(summary_llm=build_summary_llm())
        self.state_store = SessionStateStore()
        self.ppt_record_store = PptRecordStore()
        self.intent_router = IntentRouterService(self.llm)
        self.chat_responder = ChatResponder(self.llm)
        self.recursion_limit = settings.agent_multi_recursion_limit

        # build_delegation_tools() 会为每个 Agent Role 分别调用一次
        # build_delegation_tool()：先通过 build_specialist_agent() 创建对应的
        # Specialist Agent，再定义一个闭包函数 delegate_to_specialist()，由
        # @tool 包装成 Master 可调用的 Delegation Tool。每次调用都有独立的
        # 函数作用域，返回的 Tool 会一直持有本次创建的 Specialist Agent 引用，
        # 因此各 Specialist Agents 不会互相覆盖；只要该 Tool 仍保存在
        # self.delegation_tools 中，对应的 Specialist Agent 实例就会继续存在。
        self.delegation_tools = build_delegation_tools(self.llm)

        self.agent = create_agent(
            model=self.llm,
            tools=self.delegation_tools,
            system_prompt=SYSTEM_PROMPT,
            name=AGENT_NAME,
        )

    async def initialize(self) -> None:
        await self.intent_router.initialize()
        logger.info(
            "Master Agent 初始化完成: delegation_tools=%s, recursion_limit=%s",
            [tool.name for tool in self.delegation_tools],
            self.recursion_limit,
        )

    async def _prepare_ppt_record(
        self,
        *,
        intent: str,
        session_id: str,
        session_state,
        requested_ppt_id: str | None,
        style: str | None,
    ) -> tuple[PptRecord | None, str | None]:
        """让备用 Subagents 模式也遵守 Session 引用 + PptRecord 模型。"""
        if intent == "create":
            record = PptRecord(
                ppt_id=uuid4().hex,
                style=style or "business",
                status="planning",
            )
            await self.ppt_record_store.save(record)
            await self.state_store.add_ppt(session_id, record.ppt_id)
            return record, None

        target_ppt_id = requested_ppt_id or session_state.active_ppt_id
        if not target_ppt_id:
            return None, "当前会话还没有可编辑的 PPT。"
        if target_ppt_id not in session_state.ppt_ids:
            return None, "指定的 PPT 不属于当前会话，无法编辑。"
        record = await self.ppt_record_store.load(target_ppt_id)
        if record is None:
            return None, "目标 PPT 记录不存在或无法读取。"
        await self.state_store.set_active_ppt(session_id, target_ppt_id)
        return record, None

    async def run(
        self,
        user_message: str,
        session_id: str | None = None,
        requested_action: str | None = None,
        style: str | None = None,
        ppt_id: str | None = None,
    ) -> str:
        """非流式执行一次完整的 Master → Subagents 协作。"""
        session_id = session_id or "default"
        logger.info(
            "[Master Agent] 会话 %s 收到消息: %s",
            session_id,
            user_message[:100],
        )

        # 跨请求历史由 SummaryBufferMemory 保存；create_agent 只负责维护本次
        # ainvoke 内部的 Master/Tool/Subagent 消息状态。
        history = await self.memory.load(session_id)
        session_state = await self.state_store.load(session_id)
        decision = await self.intent_router.route(
            RouteContext(
                user_message=user_message,
                requested_action=requested_action,
                active_ppt_id=session_state.active_ppt_id,
                style=style or "business",
                recent_messages=history,
            )
        )

        if decision.intent == "chat" or not decision.execute:
            reply = await self.chat_responder.invoke(user_message, history)
            await self.memory.save(session_id, user_message, reply)
            return reply

        record, context_error = await self._prepare_ppt_record(
            intent=decision.intent,
            session_id=session_id,
            session_state=session_state,
            requested_ppt_id=ppt_id,
            style=style,
        )
        if context_error or record is None:
            reply = f"无法编辑 PPT：{context_error}"
            await self.memory.save(session_id, user_message, reply)
            return reply
        messages = [
            *history,
            _build_business_context(record, decision.intent),
            HumanMessage(content=user_message),
        ]

        try:
            result = await self.agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": self.recursion_limit},
            )
        except GraphRecursionError:
            logger.exception(
                "[Master Agent] 会话 %s 超出最大递归步数",
                session_id,
            )
            return "抱歉，多智能体任务已达到最大执行步数，请简化需求后重试。"

        reply = _extract_final_reply(result, input_message_count=len(messages))
        filename = _extract_ppt_filename(result)
        if filename:
            record = await self.ppt_record_store.patch(
                record.ppt_id,
                filename=filename,
                status="completed",
            )
        else:
            await self.ppt_record_store.patch(record.ppt_id, status="completed")
        await self.memory.save(session_id, user_message, reply)
        logger.info("[Master Agent] 会话 %s 执行完成", session_id)
        return reply

    async def _stream(
        self,
        user_message: str,
        session_id: str | None,
        on_ppt_created: Callable[[str], Awaitable[None]] | None,
        requested_action: str | None,
        style: str | None,
        requested_ppt_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """流式执行的公共实现，由两个公开流式入口复用。"""
        session_id = session_id or "default"
        started_at = time.monotonic()
        response_parts: list[str] = []

        logger.info(
            "[Master Agent/流式] 会话 %s 收到消息: %s",
            session_id,
            user_message[:100],
        )
        yield make_event(
            AGENT_THINKING,
            {"message": "正在识别请求意图..."},
        )

        try:
            history = await self.memory.load(session_id)
            session_state = await self.state_store.load(session_id)
            decision = await self.intent_router.route(
                RouteContext(
                    user_message=user_message,
                    requested_action=requested_action,
                    active_ppt_id=session_state.active_ppt_id,
                    style=style or "business",
                    recent_messages=history,
                )
            )
            yield make_event(
                INTENT_ROUTED,
                {
                    "intent": decision.intent,
                    "execute": decision.execute,
                    "source": decision.source,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
            )

            if decision.intent == "chat" or not decision.execute:
                async for content in self.chat_responder.stream(user_message, history):
                    response_parts.append(content)
                    yield make_event(TEXT_DELTA, {"content": content})
            else:
                record, context_error = await self._prepare_ppt_record(
                    intent=decision.intent,
                    session_id=session_id,
                    session_state=session_state,
                    requested_ppt_id=requested_ppt_id,
                    style=style,
                )
                if context_error or record is None:
                    content = f"无法编辑 PPT：{context_error}"
                    response_parts.append(content)
                    yield make_event(TEXT_DELTA, {"content": content})
                    yield make_event(
                        DONE,
                        {
                            "session_id": session_id,
                            "elapsed_seconds": round(time.monotonic() - started_at, 2),
                        },
                    )
                    return
                messages = [
                    *history,
                    _build_business_context(record, decision.intent),
                    HumanMessage(content=user_message),
                ]

                async def persist_created_ppt(filename: str) -> None:
                    await self.ppt_record_store.patch(
                        record.ppt_id,
                        filename=filename,
                        status="completed",
                    )
                    if on_ppt_created is not None:
                        await on_ppt_created(filename)

                # streaming.py 负责把 astream_events 原始事件转换为项目业务事件；
                # Master 入口负责转发事件，并收集最终文字用于保存会话记忆。
                async for event in stream_subagent_events(
                    master=self.agent,
                    messages=messages,
                    recursion_limit=self.recursion_limit,
                    on_ppt_created=persist_created_ppt,
                ):
                    if event["event"] == TEXT_DELTA:
                        response_parts.append(event["data"].get("content", ""))
                    yield event
                await self.ppt_record_store.patch(record.ppt_id, status="completed")

        except GraphRecursionError:
            logger.exception(
                "[Master Agent/流式] 会话 %s 超出最大递归步数",
                session_id,
            )
            yield make_event(
                ERROR,
                {"message": "多智能体任务已达到最大执行步数，请简化需求后重试。"},
            )
        except Exception as exc:
            # 流式接口已经开始返回响应，无法再改用普通 HTTP 错误响应，因此将
            # 运行期异常转换为 ERROR 事件；完整堆栈仍写入服务端日志。
            logger.exception(
                "[Master Agent/流式] 会话 %s 执行异常",
                session_id,
            )
            yield make_event(ERROR, {"message": f"执行异常: {exc}"})
        finally:
            response_text = "".join(response_parts).strip()
            if response_text:
                try:
                    await self.memory.save(session_id, user_message, response_text)
                except Exception:
                    # 记忆保存失败不影响已经推送给用户的模型回复。
                    logger.exception(
                        "[Master Agent/流式] 会话 %s 保存记忆失败",
                        session_id,
                    )

            elapsed_seconds = round(time.monotonic() - started_at, 2)
            logger.info(
                "[Master Agent/流式] 会话 %s 完成, 耗时 %.2fs",
                session_id,
                elapsed_seconds,
            )

        # 不把 yield 放进 finally：如果浏览器主动断开连接，Python 会关闭这个
        # async generator，此时在 finally 中继续 yield 会引发 GeneratorExit 问题。
        # 正常完成或已转换成 ERROR 事件时才会执行到这里并发送 DONE。
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
        requested_action: str | None = None,
        style: str | None = None,
        ppt_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """普通对话的流式入口。"""
        async for event in self._stream(
            user_message=user_message,
            session_id=session_id,
            on_ppt_created=None,
            requested_action=requested_action,
            style=style,
            requested_ppt_id=ppt_id,
        ):
            yield event

    async def clear_history(self, session_id: str) -> None:
        """清除指定会话的近期记录和远期摘要。"""
        await self.memory.clear(session_id)
        logger.info("[Master Agent] 会话 %s 历史已清除", session_id)
