"""PPT 创作智能体 - 基于 ReAct + Plan-Execute 的单智能体架构。

整体流程：

    用户请求
        ↓
    从 Redis 加载当前 session 的会话记忆
        ↓
    Planner 判断任务是简单还是复杂
        ├── 简单任务：直接调用 LLM，必要时允许一轮工具调用
        │
        └── 复杂任务：Planner 生成结构化执行计划
                ↓
            按顺序执行每个计划步骤
                ↓
            每个步骤内运行 ReAct 循环：
            Think（LLM 决策）
                → Act（调用工具）
                → Observe（将工具结果写入 ToolMessage）
                → 携带更新后的 messages 继续 Think
                ↓
            所有步骤完成后，审查并整理最终结果
        ↓
    将本轮用户问题和最终 AI 回复保存到 Redis

两层循环：

- Plan-Execute 是外层业务编排：任务分类 → 创建计划 → 逐步执行 → 审查结果。
- ReAct 是每个步骤内的工具循环：Think → Act → Observe → Think。
- 本实现是单智能体：Planner 是规划组件，不是另一个能独立执行任务的 Agent。

与当前 langchain_agent.py 中“基础 create_agent()”方案的区别：

LangChain create_agent() 已经封装的通用能力：

- 调用已绑定工具的 LLM；
- 识别 AIMessage 中的 tool_calls；
- 按工具名称执行对应工具；
- 把工具结果封装成 ToolMessage 并追加到运行状态；
- 携带更新后的消息继续调用 LLM，直到产生最终回复。
- 模型可根据请求自行决定直接回答、调用一次工具，或连续调用多次工具，
  因此基础 create_agent() 同样能处理简单任务和许多复杂任务。

本文件的特点不是“LangChain Agent 做不到这些事”，而是将下列策略
作为业务规则显式写在 Agent 编排层中：

1. 任务分类：由 Planner 判断是否需要进入复杂任务流程。
2. 显式规划：先生成包含 description 和 expected_tools 的结构化步骤。
3. 分步执行：为每个计划步骤注入独立的执行上下文。
4. 可控的 ReAct 循环：手动维护 AIMessage、ToolMessage 和最大迭代次数。
5. 异常降级：单个计划步骤或第三方工具失败时，尽量不中断整个任务。
6. 最终审查：所有步骤完成后，再调用模型根据原始目标整理结果。
7. 显式分流：简单请求确定性地跳过规划和审查，减少模型调用和延迟。

基础 create_agent() 中，复杂任务通常由模型在 ReAct 循环里边执行边决定下一步，
这是“隐式规划”；它并不默认先调用一个独立 Planner，也不保证每次都产生
本项目这种固定 JSON 步骤列表。

如果使用 LangChain 也需要显式计划，可以在创建 Agent 时加入 TodoListMiddleware，
让 Agent 在运行中维护结构化待办计划；或者使用自定义 LangGraph，把
“分类 → 规划 → 执行 → 审查”写成独立节点。这些能力通常在 create_agent()
时通过 middleware 配置，或在 Graph 定义中编排，并不要求业务代码在每次
ainvoke() 之前额外手动调用一个固定的规划函数。

因此，这里比较的只是本项目的两个当前实现：
“未配置规划 middleware 的基础 create_agent() 版本”和“显式手写 Plan-Execute 的 PPTAgent 版本”。
"""

import logging
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from openai import APIError

from app.agent.planner import Planner
from app.agent.prompts import REVIEW_PROMPT, SYSTEM_PROMPT
from app.core.config import settings
from app.tools import all_tools
from app.context import SummaryBufferMemory

logger = logging.getLogger(__name__)

DEFAULT_MAX_REACT_ITERATIONS = 10


class PPTAgent:
    """PPT 创作智能体（ReAct + Plan-Execute）"""

    def __init__(self):
        # 1. 主 LLM（带工具绑定）
        self.llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.7,
        )

        # 2. 摘要 LLM（低 temperature，用于记忆摘要）
        summary_llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.0,
        )

        # 3. 工具
        self.tools = all_tools
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 4. 组件
        self.memory = SummaryBufferMemory(summary_llm=summary_llm)
        self.planner = Planner()

        # 5. 配置
        self.max_react_iterations = getattr(
            settings, "agent_max_react_iterations", DEFAULT_MAX_REACT_ITERATIONS
        )
        self.enable_review = getattr(settings, "agent_enable_plan_review", True)

    # ================================================================
    # 主入口
    # ================================================================

    async def run(
        self,
        user_message: str,
        session_id: str | None = None,
    ) -> str:
        """智能体主入口 — Plan-Execute + ReAct

        Returns:
            回复文本
        """
        session_id = session_id or "default"
        start = time.time()

        logger.info("[智能体] 会话 %s 收到消息: %s", session_id, user_message[:100])

        # 1. 加载记忆
        history = await self.memory.load(session_id)

        # 2. 任务分类
        history_summary = self._extract_summary(history)
        is_complex = await self.planner.classify(user_message, history_summary)

        if not is_complex:
            # ── 简单任务：直接回复 ──
            logger.info("[智能体] 简单任务，直接回复")
            response_text = await self._direct_response(user_message, history)
        else:
            # ── 复杂任务：Plan-Execute + ReAct ──
            logger.info("[智能体] 复杂任务，启动 Plan-Execute")

            # 3. 创建计划
            plan_steps = await self.planner.create_plan(user_message, history_summary)
            logger.info("[智能体] 计划: %d 步", len(plan_steps))

            # 4. 逐步执行（每步 ReAct）
            messages = self._build_messages(user_message, history)

            for idx, step in enumerate(plan_steps):
                # 注入步骤上下文
                step_context = (
                    f"[计划执行] 当前步骤 {idx + 1}/{len(plan_steps)}\n"
                    f"步骤描述: {step['description']}\n"
                    f"预期使用的工具: {', '.join(step.get('expected_tools', [])) or '自动判断'}\n"
                    f"请执行此步骤。"
                )
                messages.append(SystemMessage(content=step_context))

                logger.info(
                    "[智能体] 步骤 %d/%d: %s",
                    idx + 1,
                    len(plan_steps),
                    step["description"][:50],
                )

                try:
                    await self._react_loop(messages)
                except APIError as exc:
                    logger.error("[智能体] 步骤 %d 执行失败: %s", idx + 1, exc)

            # 5. 审查结果
            if self.enable_review:
                response_text = await self._review_results(
                    messages, plan_steps, user_message
                )
            else:
                response_text = messages[-1].content if messages else "任务执行完成。"

        # 6. 保存记忆
        await self.memory.save(session_id, user_message, response_text)

        logger.info(
            "[智能体] 会话 %s 完成, 耗时 %.2fs",
            session_id,
            time.time() - start,
        )
        return response_text

    # ================================================================
    # ReAct 循环
    # ================================================================

    async def _react_loop(self, messages: list) -> str:
        """ReAct 循环：Think → Act → Observe

        Args:
            messages: 消息列表（原地修改）

        Returns:
            最终回复文本
        """
        for iteration in range(self.max_react_iterations):
            # THINK
            response = await self.llm_with_tools.ainvoke(messages)

            # 无工具调用 → 最终回复
            if not response.tool_calls:
                logger.debug("[ReAct] 迭代 %d: 最终回复", iteration + 1)
                return response.content or ""

            # ACT：执行工具
            messages.append(response)
            for tool_call in response.tool_calls:
                logger.info(
                    "[ReAct] 迭代 %d: 调用 %s(%s)",
                    iteration + 1,
                    tool_call["name"],
                    str(tool_call["args"])[:100],
                )
                observation = await self._execute_tool(
                    tool_call["name"], tool_call["args"]
                )
                # OBSERVE
                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"],
                    )
                )

        logger.warning("[ReAct] 达到最大迭代 %d", self.max_react_iterations)
        return "达到最大推理迭代次数，已部分完成。如需继续请告诉我。"

    # ================================================================
    # 简单任务
    # ================================================================

    async def _direct_response(self, user_message: str, history: list) -> str:
        """简单任务：直接对话（允许一轮工具调用）"""
        messages = self._build_messages(user_message, history)
        response = await self.llm_with_tools.ainvoke(messages)

        if response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                result = await self._execute_tool(tc["name"], tc["args"])
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tc["id"],
                    )
                )
            response = await self.llm_with_tools.ainvoke(messages)

        return response.content

    # ================================================================
    # 工具执行
    # ================================================================

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行单个工具"""
        tool = self.tool_map.get(tool_name)
        if not tool:
            return f"错误：未知工具 '{tool_name}'"
        try:
            result = await tool.ainvoke(tool_args)
            return str(result) if result is not None else "工具执行完成（无返回值）"
        # 工具可来自任意第三方集成，此处是防止单个工具中断 Agent 的隔离边界。
        except Exception as exc:
            logger.exception("工具执行失败: %s", tool_name)
            return f"工具执行失败: {exc}"

    # ================================================================
    # 审查结果
    # ================================================================

    async def _review_results(
        self, messages: list, plan_steps: list[dict], goal: str
    ) -> str:
        """审查计划执行结果，生成最终回复"""
        lines = [f"原始目标: {goal}", f"计划步骤: {len(plan_steps)} 个", ""]
        for i, step in enumerate(plan_steps):
            lines.append(f"  ✅ 步骤 {i + 1}: {step['description']}")
        plan_summary = "\n".join(lines)

        review_text = REVIEW_PROMPT.format(plan_summary=plan_summary)
        messages.append(SystemMessage(content=review_text))

        try:
            response = await self.llm_with_tools.ainvoke(messages)
            return str(response.text)
        except APIError as exc:
            logger.error("审查失败: %s", exc)
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    reply_text = str(message.text).strip()
                    if reply_text:
                        return reply_text
            return "任务执行完成。"

    # ================================================================
    # 辅助方法
    # ================================================================

    def _build_messages(self, user_message: str, history: list) -> list:
        """构建消息列表"""
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(history)
        messages.append(HumanMessage(content=user_message))
        return messages

    @staticmethod
    def _extract_summary(history: list) -> str:
        """从历史消息中提取摘要"""
        for msg in history:
            if isinstance(msg, SystemMessage) and "对话历史摘要" in msg.content:
                return msg.content
        return ""

    async def clear_history(self, session_id: str):
        """清除会话历史"""
        await self.memory.clear(session_id)
        logger.info("[智能体] 会话 %s 历史已清除", session_id)
