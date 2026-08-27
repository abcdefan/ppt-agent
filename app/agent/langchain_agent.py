"""LangChain Agent - 基于 langchain 的 ReAct 智能体"""

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from redis.exceptions import RedisError

from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import settings
from app.schemas.events import (
    AGENT_THINKING,
    DONE,
    ERROR,
    TEXT_DELTA,
    TOOL_CALL,
    TOOL_RESULT,
    make_event,
)
from app.tools import all_tools
from app.context import SummaryBufferMemory

logger = logging.getLogger(__name__)

# SSE 工具结果展示的最大长度（超出截断，避免单个事件过大）
TOOL_RESULT_PREVIEW_LEN = 500


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_LEN) -> str:
    """截断过长的工具结果，供 SSE 事件展示"""
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + "...(已截断)"


class LangChainAgent:
    """LangChain 智能体"""

    def __init__(self):
        # 1. LLM
        self.llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.7,
        )

        # 2. 摘要 LLM
        summary_llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.0,
        )

        # 3. 记忆组件
        self.memory = SummaryBufferMemory(summary_llm=summary_llm)

        # 4. 创建 ReAct agent
        #    create_agent 内部自动处理 Think → Act → Observe 循环
        self.agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=SYSTEM_PROMPT,
        )

        logger.info("LangChain Agent 初始化完成 (langchain create_agent)")

    async def run(
        self,
        user_message: str,
        session_id: str | None = None,
    ) -> str:
        """运行 LangChain 智能体

        Args:
            user_message: 用户消息
            session_id: 会话 ID

        Returns:
            回复文本
        """
        # 每次 run() 都是一次独立的请求，整体流程如下：
        #
        #   每次调用 run()
        #       ↓
        #   创建一个新的局部 messages 列表
        #       ↓
        #   从 Redis 加载该 session 的历史
        #       ↓
        #   加入当前用户消息
        #       ↓
        #   交给 LangGraph Agent 执行
        #       ↓
        #   Agent 在本次 ainvoke 状态中不断追加 AIMessage / ToolMessage
        #       ↓
        #   提取最终 AI 回复
        #       ↓
        #   只把“当前用户问题 + 最终 AI 回复”追加到 Redis
        #       ↓
        #   本次局部 messages 生命周期结束
        #
        # 因此，messages 不是跨请求共享的全局变量。下一轮请求会
        # 重新创建列表，并使用 session_id 从 Redis 重建对话上下文。
        session_id = session_id or "default"

        logger.info(
            "[LangChain Agent] 会话 %s 收到消息: %s",
            session_id,
            user_message[:100],
        )

        # 1. 加载记忆
        history = await self.memory.load(session_id)

        # 2. 构建消息列表
        messages = []
        messages.extend(history)
        messages.append(HumanMessage(content=user_message))

        # 3. 调用 LangGraph Agent，执行一次完整的 Agent 运行。
        #
        # create_agent() 返回的是一个已编译的 LangGraph，ainvoke() 会把
        # messages 作为这次运行的初始状态，并在内部自动执行 ReAct 循环：
        #   1. 把当前消息发给 LLM；
        #   2. 如果 LLM 返回 tool_calls，根据工具名称执行对应工具；
        #   3. 把 AIMessage 和工具结果 ToolMessage 追加到消息状态；
        #   4. 携带更新后的全部消息再次调用 LLM；
        #   5. 直到 LLM 不再请求工具，生成最终文字回复。
        #
        # 这里自动维护的是“本次 ainvoke 运行期间”的上下文。
        # 跨用户请求的历史不是由此 Agent 自动持久化的，而是由上面的
        # self.memory.load() 读取，再由下面的 self.memory.save() 保存。
        result = await self.agent.ainvoke(
            {"messages": messages},
        )

        # 4. 从 Agent 的最终状态中提取给用户展示的文字回复。
        #
        # result 是 LangGraph 返回的状态字典，result["messages"] 通常包含：
        #   - 调用前传入的历史消息和本次 HumanMessage；
        #   - ReAct 过程中产生的 AIMessage（可能只包含 tool_calls）；
        #   - 每次工具执行后产生的 ToolMessage；
        #   - Agent 最后生成的文字 AIMessage。
        #
        # result["messages"] 的前半部分是本次调用前传入的 messages，
        # 从 len(messages) 开始的部分才是本次 ainvoke 新生成的消息。
        # 这里只检查新增消息，避免本次没有生成最终文本时，误把历史中的
        # 某条 AI 回复当成本次回复返回给用户。
        final_reply = ""
        result_messages = result.get("messages", [])
        generated_messages = result_messages[len(messages) :]

        # 最终回复通常位于新增消息的末尾，因此从后向前查找；找到第一条
        # 符合条件的消息后即可停止。
        for msg in reversed(generated_messages):
            # 只接受 AIMessage，并排除仍携带 tool_calls 的中间消息。
            # 即使这类中间消息同时带有文字，它的主要作用仍是请求执行工具，
            # 不能将它误认为 Agent 已经生成的最终答案。
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                # msg.content 可能是字符串，也可能是文本/图像等内容块列表。
                # msg.text 会由 LangChain 统一提取其中的文本；strip() 用于
                # 过滤空字符串和只含空白字符的消息。
                reply_text = str(msg.text).strip()
                if reply_text:
                    # 从后向前找到的第一条非空 AI 文本，就是本次运行需要
                    # 返回给用户并保存到长期记忆中的最终回复。
                    final_reply = reply_text
                    break

        # 正常情况下 Agent 都会生成最终 AI 文本。如果消息列表为空，
        # 或只有 tool_calls / ToolMessage 而没有文字，则使用兜底提示，
        # 保证 run() 始终返回一个可展示的非空字符串。
        if not final_reply:
            final_reply = "任务执行完成，但未生成文字回复。"

        # 5. 保存记忆
        await self.memory.save(session_id, user_message, final_reply)

        logger.info("[LangChain Agent] 会话 %s 完成", session_id)
        return final_reply

    # ================================================================
    # 流式入口（SSE）— 基于 langgraph agent 的 astream_events
    # ================================================================

    async def _stream_core(
        self,
        user_message: str,
        session_id: None | str = None,
        on_ppt_created: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict]:
        """流式核心实现 — 基于 langgraph agent 的 astream_events(version="v2")。

        对外不要直接调用，请用 run_stream（基础对话）或 run_ppt_stream（PPT 创作）。

        将底层 Runnable 事件映射为 SSE 业务事件：
        - on_chat_model_stream → TEXT_DELTA（逐 token）
        - on_tool_start        → TOOL_CALL
        - on_tool_end          → TOOL_RESULT

        Args:
            on_ppt_created: 可选异步回调。当 generate_ppt 工具成功执行时，
                以生成的文件名调用，由控制器完成配额扣减与记录（agent 不碰 DB）。

        Yields:
            {"event": <事件类型>, "data": <dict>} — 由控制器转为 SSE 格式
        """
        session_id = session_id or "default"
        start = time.time()
        response_parts: list[str] = []

        logger.info(
            "[LangChain Agent/流式] 会话 %s 收到消息: %s",
            session_id,
            user_message[:100],
        )

        try:
            # 1. 加载记忆
            yield make_event(AGENT_THINKING, {"message": "正在加载对话历史..."})
            history = await self.memory.load(session_id)

            # 2. 构建消息
            messages = list(history)
            messages.append(HumanMessage(content=user_message))

            yield make_event(AGENT_THINKING, {"message": "正在思考..."})

            # 3. 订阅 agent 事件流（自动 ReAct 循环）
            async for ev in self.agent.astream_events(
                {"messages": messages}, version="v2"
            ):
                kind = ev["event"]

                if kind == "on_chat_model_stream":
                    chunk = ev["data"].get("chunk")
                    content = getattr(chunk, "content", "") if chunk else ""
                    if content:
                        response_parts.append(content)
                        yield make_event(TEXT_DELTA, {"content": content})

                elif kind == "on_tool_start":
                    tool_input = ev["data"].get("input")
                    yield make_event(
                        TOOL_CALL,
                        {
                            "tool": ev["name"],
                            "args": tool_input
                            if isinstance(tool_input, dict)
                            else {"input": tool_input},
                        },
                    )

                elif kind == "on_tool_end":
                    output = ev["data"].get("output")
                    # output 通常是 ToolMessage，取其 content；否则直接字符串化
                    content = getattr(output, "content", None)
                    result_str = (
                        content
                        if content
                        else (str(output) if output is not None else "")
                    )
                    yield make_event(
                        TOOL_RESULT,
                        {"tool": ev["name"], "result": _truncate(result_str)},
                    )

                    # generate_ppt 成功 → 触发回调（控制器侧完成扣减+记录）
                    if ev["name"] == "generate_ppt" and on_ppt_created is not None:
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
                                "[LangChain Agent/流式] generate_ppt 结果解析失败: %s",
                                e,
                            )
        except (
            OpenAIError,
            RedisError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ) as e:
            # 只把可预期的运行期故障转换成 SSE 错误事件；未知的代码缺陷
            # 不在这里盲目吞掉，而是继续向上传播，便于及时发现和修复。
            logger.exception("[LangChain Agent/流式] 执行异常: %s", e)
            yield make_event(ERROR, {"message": f"执行异常: {e}"})
        finally:
            # 4. 保存记忆（失败不影响已推送的事件）
            response_text = "".join(response_parts)
            if response_text:
                try:
                    await self.memory.save(session_id, user_message, response_text)
                except (
                    OpenAIError,
                    RedisError,
                    json.JSONDecodeError,
                    TypeError,
                ) as e:
                    # 保存失败不影响已经推送给客户端的流式内容，但需要保留
                    # 完整堆栈，方便定位 Redis、摘要模型或历史数据格式问题。
                    logger.exception("[LangChain Agent/流式] 保存记忆失败: %s", e)
            logger.info(
                "[LangChain Agent/流式] 会话 %s 完成, 耗时 %.2fs",
                session_id,
                time.time() - start,
            )
            yield make_event(
                DONE,
                {
                    "session_id": session_id,
                    "elapsed_seconds": round(time.time() - start, 2),
                },
            )

    async def run_stream(
        self,
        user_message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """对话流式响应"""
        async for event in self._stream_core(user_message, session_id):
            yield event

    async def run_ppt_stream(
        self,
        user_message: str,
        session_id: str | None = None,
        on_ppt_created: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict]:
        """PPT 创作流式（供 ppt 控制器调用）。

        generate_ppt 成功时触发 on_ppt_created 回调，
        由控制器完成配额扣减与创作记录。
        """
        async for event in self._stream_core(
            user_message, session_id, on_ppt_created=on_ppt_created
        ):
            yield event

    async def clear_history(self, session_id: str):
        """清除会话历史"""
        await self.memory.clear(session_id)
        logger.info("[LangChain Agent] 会话 %s 历史已清除", session_id)
