"""会话记忆组件 - SummaryBufferMemory（混合摘要记忆）。"""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core import redis as redis_module

logger = logging.getLogger(__name__)

# Redis key 前缀
_HISTORY_PREFIX = "chat:history:"  # 近期完整对话
_SUMMARY_PREFIX = "chat:summary:"  # 远期对话摘要

# 摘要提示词
_SUMMARY_PROMPT = """请逐步分析以下对话记录，用一段简洁的中文摘要概括其中的关键信息。
重点保留：用户的个人信息、偏好、核心需求、已确定的结论等重要上下文。
忽略：寒暄、重复提问、无关紧要的细节。

对话记录：
{conversation}

请直接输出摘要内容，不要加前缀说明："""


class SummaryBufferMemory:
    """混合摘要记忆组件"""

    def __init__(
        self,
        summary_llm: BaseChatModel,
        buffer_rounds: int = 5,
        summary_trigger: int = 7,
    ):
        self.summary_llm = summary_llm
        self.buffer_rounds = buffer_rounds
        self.summary_trigger = summary_trigger

    async def load(self, session_id: str) -> list:
        """加载一个会话可供模型继续使用的完整上下文。

        方法先从 Redis 读取远期对话摘要，并将其转换成一条
        ``SystemMessage``；随后读取仍被完整保留的近期对话，按照角色依次
        还原成 ``HumanMessage`` 和 ``AIMessage``。最终顺序为“远期摘要在前，
        近期原文在后”，调用方可以直接把结果加入下一次模型请求。

        典型调用时机是每轮模型回答之前，流程如下：

        用户发送本轮新消息
        ↓
        load() 取出长期摘要 + 近期完整对话
        ↓
        拼接本轮用户消息后交给模型生成回复
        ↓
        调用 save() 保存本轮用户消息和 AI 回复

        Args:
            session_id: 会话的唯一标识，用于拼接该会话在 Redis 中的 key。

        Returns:
            按时间顺序排列的 LangChain 消息列表。Redis 客户端不可用，或者
            当前会话尚无任何记忆时，返回空列表。
        """
        if not redis_module.redis_client:
            return []

        messages = []

        # 1. 远期摘要 → SystemMessage
        summary = await redis_module.redis_client.get(f"{_SUMMARY_PREFIX}{session_id}")
        if summary:
            messages.append(SystemMessage(content=f"[对话历史摘要]\n{summary}"))

        # 2. 近期完整对话 → HumanMessage / AIMessage
        raw = await redis_module.redis_client.get(f"{_HISTORY_PREFIX}{session_id}")
        if raw:
            for record in json.loads(raw):
                if record["role"] == "human":
                    messages.append(HumanMessage(content=record["content"]))
                elif record["role"] == "ai":
                    messages.append(AIMessage(content=record["content"]))

        return messages

    async def _summarize(self, summary_key: str, old_records: list[dict]):
        """将移出近期缓冲区的早期对话合并到长期摘要中。

        如果 Redis 中已经存在摘要，会把旧摘要和本次待压缩的对话一起交给
        ``summary_llm``，让模型生成一份覆盖全部远期上下文的新摘要，而不是
        简单拼接多段摘要。生成完成后，新摘要会覆盖写回 ``summary_key``。

        这是 ``save()`` 的内部辅助方法；调用前已经确认 Redis 客户端可用。

        Args:
            summary_key: 当前会话长期摘要对应的完整 Redis key。
            old_records: 需要从近期原文中移除并压缩的消息记录。每条记录包含
                ``role`` 和 ``content`` 字段。

        Returns:
            无返回值；摘要结果直接写入 Redis。
        """
        # 1. 读取以前已经生成的长期摘要。
        # 第一次触发摘要时 Redis 中还没有这个 key，因此使用空字符串作为默认值。
        existing = await redis_module.redis_client.get(summary_key) or ""

        # 2. 组装本次需要交给摘要模型的全部材料。
        # parts 最终包含两部分：以前的长期摘要（如果存在）以及 save() 本次从
        # 近期 history 中切出来的 old_records。这样模型可以在保留旧摘要信息的
        # 同时，把新移出的早期对话继续合并进去。
        parts: list[str] = []
        if existing:
            # 旧摘要不是最终直接拼接到 Redis，而是作为模型本次重新摘要的输入。
            parts.append(f"[已有历史摘要]\n{existing}\n\n[新增长对话]")

        # old_records 中一轮对话对应两条记录：human 一条、ai 一条。
        # 这里将内部存储的角色名称转换成更容易让模型理解的“用户”和“AI”。
        for r in old_records:
            role = "用户" if r["role"] == "human" else "AI"
            parts.append(f"{role}: {r['content']}")

        # 3. 用换行把“旧摘要 + 本次移出的旧对话”合并为一段完整文本。
        conversation_text = "\n".join(parts)

        # 核心理解：把要压缩的对话内容填入摘要 Prompt，再将整段 Prompt 作为
        # 一条用户消息发送给 AI 并等待回复；AI 的回复就是本次生成的新摘要。

        # 4. 使用 Python 的 str.format() 替换提示词中的 {conversation} 占位符。
        # 这一步还没有调用模型，只是生成一个普通字符串。最终字符串等价于：
        # “请压缩以下对话…… + 旧摘要 + 本次移出的旧对话”。
        rendered_prompt = _SUMMARY_PROMPT.format(conversation=conversation_text)

        # 5. ChatModel 接收的是消息列表，所以需要把完整提示词包装成
        # HumanMessage，再放入 list 中。这里只发送一条消息，但仍使用列表，
        # 是因为聊天模型统一支持多轮的 SystemMessage/HumanMessage/AIMessage。
        summary_messages = [HumanMessage(content=rendered_prompt)]

        # 6. ainvoke() 是 LangChain 对聊天模型的一次“通用异步调用”，并不是
        # 专门用于摘要的函数。模型之所以返回摘要，是因为 rendered_prompt 明确
        # 要求它压缩对话；换成问答提示词，同一个 ainvoke() 就会执行普通问答。
        # await 会等待模型响应，response.content 就是模型本次生成的摘要文本。
        response = await self.summary_llm.ainvoke(summary_messages)

        # 7. 用合并后的新摘要覆盖 Redis 中原来的 summary_key。
        # 这是“逻辑上追加记忆、物理上覆盖摘要”：旧摘要已经作为输入参与了本次
        # 压缩，所以新摘要理论上同时包含旧信息和本次新增的远期对话。
        # 注意：这里没有修改 history_key；save() 会在本方法返回后，将保留下来的
        # 最近 buffer_rounds 轮完整原文写回 history_key。
        await redis_module.redis_client.set(summary_key, response.content)

    async def save(self, session_id: str, user_msg: str, ai_msg: str):
        """保存一轮用户与模型的对话，并在需要时压缩早期记录。

        一轮对话由一条用户消息和一条 AI 回复组成。方法会先把这两条消息
        追加到近期历史；当总轮数严格大于 ``summary_trigger`` 时，把较早的
        轮次交给 ``_summarize()``，只留下最近 ``buffer_rounds`` 轮完整原文，
        最后把保留的近期记录写回 Redis。

        Args:
            session_id: 会话的唯一标识，用于区分不同用户或不同对话。
            user_msg: 本轮用户发送的原始消息。
            ai_msg: 本轮模型生成的回复。

        Returns:
            无返回值。Redis 客户端不可用时直接结束，不保存任何数据。
        """
        if not redis_module.redis_client:
            return

        history_key = f"{_HISTORY_PREFIX}{session_id}"
        summary_key = f"{_SUMMARY_PREFIX}{session_id}"

        # 1. 读取现有近期对话，追加本轮
        raw = await redis_module.redis_client.get(history_key)
        records: list[dict] = json.loads(raw) if raw else []
        records.append({"role": "human", "content": user_msg})
        records.append({"role": "ai", "content": ai_msg})

        # 2. 判断是否触发摘要
        total_rounds = len(records) // 2
        if total_rounds > self.summary_trigger:
            rounds_to_summarize = total_rounds - self.buffer_rounds
            cut_index = rounds_to_summarize * 2

            old_records = records[:cut_index]
            records = records[cut_index:]

            await self._summarize(summary_key, old_records)

        # 3. 写回 Redis
        await redis_module.redis_client.set(
            history_key,
            json.dumps(records, ensure_ascii=False),
        )
        # 摘要 key 保持与 history key 相同的存活状态（不过期）

    async def clear(self, session_id: str):
        """永久清除一个会话保存在 Redis 中的全部记忆。

        该操作会同时删除近期完整对话和远期摘要两个 key，使后续 ``load()``
        返回空列表。Redis 的 ``delete`` 可以安全处理不存在的 key，因此重复
        清除同一个会话不会报错。

        Args:
            session_id: 需要清除记忆的会话唯一标识。

        Returns:
            无返回值。Redis 客户端不可用时直接结束。
        """
        if not redis_module.redis_client:
            return
        await redis_module.redis_client.delete(
            f"{_HISTORY_PREFIX}{session_id}",
            f"{_SUMMARY_PREFIX}{session_id}",
        )
