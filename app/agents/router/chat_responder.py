"""不绑定 Tools 的普通对话响应器。"""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

CHAT_REPLY_PROMPT = """你是 PPTCreator 助手，当前处于不执行文件操作的对话分支。
请使用简洁、自然的中文直接回答。不要调用工具，不要生成或修改 PPT 文件，也不要
声称已经操作文件。

用户也可能正在通过多轮对话描述 Create/Edit 需求：
- 用户明确还在继续描述时，简短确认已经理解并请其继续，不要催促执行；
- 用户表示需求已经说完但没有明确要求开始时，简要总结并询问是否开始；
- 用户只是讨论方案或询问问题时，正常提供建议；
- 用户要求不明确时，可以自然询问缺少的信息。
不要要求用户必须使用固定口令或前端模式按钮。
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
