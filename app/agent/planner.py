"""任务分类与计划创建器 - 负责判断任务复杂度并生成执行计划"""

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIError
from pydantic import ValidationError

from app.agent.prompts import CLASSIFY_PROMPT, PLAN_PROMPT
from app.core.config import settings
from app.schemas.structured_output import ExecutionPlan, TaskClassification

logger = logging.getLogger(__name__)


class Planner:
    """任务分类与计划创建器（使用低温 LLM，无工具绑定）"""

    def __init__(self):
        # 专用低温 LLM，用于分类和规划（不绑定工具）
        self.llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0.0,
        )

        self.classify_llm = self.llm.with_structured_output(
            TaskClassification, method="json_mode"
        )

        self.plan_llm = self.llm.with_structured_output(
            ExecutionPlan, method="json_mode"
        )

    async def classify(self, user_message: str, history_summary: str = "") -> bool:
        """判断任务是否为复杂任务。

        Returns:
            True 表示复杂任务（需要规划），False 表示简单任务
        """
        prompt = CLASSIFY_PROMPT.format(user_message=user_message)
        if history_summary:
            prompt += f"\n\n对话历史摘要：{history_summary}"

        try:
            result: TaskClassification = await self.classify_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            logger.info(
                "任务分类: 复杂=%s, 理由: %s",
                result.is_complex,
                result.reason or "无",
            )
            return result.is_complex
        except (APIError, OutputParserException, ValidationError, TypeError) as exc:
            logger.warning("任务分类失败，默认简单任务: %s", exc)

        return False

    async def create_plan(
        self, user_message: str, history_summary: str = ""
    ) -> list[dict]:
        """为复杂任务创建执行计划。

        Returns:
            步骤列表 [{"description": "...", "expected_tools": [...]}, ...]
        """
        max_steps = getattr(settings, "agent_max_plan_steps", 8)
        prompt = PLAN_PROMPT.format(user_message=user_message, max_steps=max_steps)
        if history_summary:
            prompt += f"\n\n对话历史摘要：{history_summary}"

        try:
            result: ExecutionPlan = await self.plan_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )

            # result 是经过 Pydantic 校验的 ExecutionPlan 对象，
            # result.steps 是 list[PlanStep]，例如：
            # [
            #     PlanStep(description="分析需求", expected_tools=[]),
            #     PlanStep(
            #         description="生成 PPT",
            #         expected_tools=["generate_ppt"],
            #     ),
            # ]
            #
            # [:max_steps] 先将计划限制在配置的最大步骤数内；
            # step.model_dump() 再把每个 PlanStep 转换成普通字典。
            # 转换后的 steps 为：
            # [
            #     {"description": "分析需求", "expected_tools": []},
            #     {
            #         "description": "生成 PPT",
            #         "expected_tools": ["generate_ppt"],
            #     },
            # ]
            # 这样可以兼容 PPTAgent 中 step["description"] 和
            # step.get("expected_tools", []) 这类字典访问方式。
            steps = [step.model_dump() for step in result.steps[:max_steps]]
            logger.info(
                "计划创建成功: %d 步, 难度=%s - %s",
                len(steps),
                result.estimated_difficulty,
                " → ".join(step["description"][:20] for step in steps),
            )
            return steps
        except (
            APIError,
            OutputParserException,
            ValidationError,
            TypeError,
        ) as exc:
            logger.warning("计划创建失败，降级为单步: %s", exc)

        # 降级：单步计划
        return [{"description": f"完成用户请求: {user_message}", "expected_tools": []}]
