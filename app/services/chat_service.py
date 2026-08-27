"""LangChain 大模型对话服务"""

import logging

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.tools import all_tools
from app.context import SummaryBufferMemory

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default"


# 系统提示词（角色设定 + 工具使用指南）
SYSTEM_PROMPT = """你是一个 AI 助手，擅长帮助用户创作 PPT。

你可以使用以下工具来完成任务：

1. **read_file** - 读取文件内容
2. **write_file** - 写入文件
3. **list_files** - 列出目录中的文件
4. **generate_ppt** - 根据结构化 JSON 数据生成 PPTX 文件

当用户请求创建 PPT 时，请按以下流程操作：
1. 使用 generate_ppt 生成 PPTX 文件，将幻灯片数据作为 slides 参数传入
2. 告知用户文件路径和生成结果

请用简洁、专业的语气回答。"""

MAX_TOOL_ITERATIONS = 5


class ChatService:
    """基于 LangChain 的对话服务"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.7,
            max_tokens=1000,
        )

        # 摘要 LLM（低 temperature 保证稳定）
        summary_llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.0,
        )
        # 记忆组件（独立抽取，其他 agent 也可复用）
        self.memory = SummaryBufferMemory(summary_llm=summary_llm)

        # 工具绑定
        self.tools = all_tools
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    async def chat(self, user_message: str, session_id: None | str = None) -> str:
        """调用大模型进行对话"""
        session_id = session_id or DEFAULT_SESSION_ID

        # 1. 读取记忆
        history = await self.memory.load(session_id)

        # 2. 注入到提示词模板
        messages = self.prompt_template.format_messages(
            history=history,
            input=user_message,
        )

        # 打印发送给 LLM 的完整消息
        logger.info(
            "[会话ID: %s] 发送给 AI 的完整消息 (共 %d 条):", session_id, len(messages)
        )
        for idx, msg in enumerate(messages):
            role_name = type(msg).__name__.replace("Message", "")
            logger.info("  [%d] %s: %s", idx, role_name, msg.content)

        # 工具调用
        needs_final_response = True
        for iteration in range(MAX_TOOL_ITERATIONS):
            # 多次调用大模型
            response = await self.llm_with_tools.ainvoke(messages)

            # 没有工具调用 → 最终响应
            if not response.tool_calls:
                needs_final_response = False
                break

            # 有工具调用 → 执行工具
            messages.append(response)
            logger.info(
                "[会话ID: %s] 第 %d 轮工具调用: %s",
                session_id,
                iteration + 1,
                [tc["name"] for tc in response.tool_calls],
            )

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # 执行工具
                tool_result = await self._execute_tool(tool_name, tool_args)

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

        # 达到最大工具调用轮数后，禁止继续调用工具并生成最终回答
        if needs_final_response:
            logger.warning(
                "[会话ID: %s] 已达到最大工具调用轮数 %d，生成最终回答",
                session_id,
                MAX_TOOL_ITERATIONS,
            )
            response = await self.llm.ainvoke(messages)

        # 4. 保存本轮对话到记忆
        await self.memory.save(session_id, user_message, response.content)

        return response.content

    async def clear_history(self, session_id: str):
        """清除指定会话的历史记录"""
        await self.memory.clear(session_id)

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行单个工具调用"""
        tool = self.tool_map.get(tool_name)
        if not tool:
            return f"错误：未知工具 '{tool_name}'"

        try:
            result = await tool.ainvoke(tool_args)
            return str(result) if result is not None else "工具执行完成（无返回值）"
        except Exception as exc:
            logger.exception("工具执行失败: %s", tool_name)
            return f"工具执行失败: {exc}"


# 模块级单例
chat_service = ChatService()
