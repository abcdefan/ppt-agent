"""不绑定 Tools 的普通对话响应器。"""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

CHAT_REPLY_PROMPT = """你是 PPTCreator 助手，当前处理普通对话或简单问答。
请使用简洁、自然的中文直接回答。不要调用工具，不要生成 PPT 文件，也不要声称已经创建文件。
如果用户明确要求制作 PPT，应提示用户重新描述需求或使用“创建 PPT”模式。
"""


class ChatResponder:
    def __init__(self, llm):
        self._llm = llm

    @staticmethod
    def _messages(user_message: str, history: list) -> list:
        return [
            SystemMessage(content=CHAT_REPLY_PROMPT),
            *history,
            HumanMessage(content=user_message),
        ]

    async def invoke(self, user_message: str, history: list) -> str:
        response: AIMessage = await self._llm.ainvoke(
            self._messages(user_message, history)
        )
        return str(response.text).strip()

    async def stream(self, user_message: str, history: list) -> AsyncIterator[str]:
        async for chunk in self._llm.astream(self._messages(user_message, history)):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content
