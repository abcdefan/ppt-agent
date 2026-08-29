"""显式信号 → Embedding → LLM 的三层意图路由。"""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI

from app.agents.router.intent_index import IntentIndex
from app.agents.router.models import (
    LLMIntentDecision,
    RouteContext,
    RouteDecision,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

INTENT_ROUTER_PROMPT = """你是 PPTCreator 的入口意图分类器。

只允许选择以下三种意图：
- chat：问候、闲聊、知识问答、解释概念或普通对话，不要求生成 PPT 文件。
- create：用户要求制作、创建、生成演示文稿、幻灯片或 PPT 文件。
- edit：用户要求修改、替换、美化、增删已有 PPT 的图片、图表或视觉样式。

结合当前用户消息和少量近期上下文判断。不要因为系统是 PPTCreator 就把普通问题判为 create。
必须返回 JSON：{"intent": "chat/create/edit", "reason": "一句话理由"}。
"""


class IntentRouterService:
    """与 Workflow/Subagents 无关的公共意图识别服务。"""

    def __init__(self, llm, embedding_client=None):
        self._structured_llm = llm.with_structured_output(
            LLMIntentDecision,
            method="json_mode",
        )
        self._embedding_client = embedding_client or self._build_embedding_client()
        self._intent_index: IntentIndex | None = None
        self._embedding_disabled = self._embedding_client is None
        self._initialize_lock = asyncio.Lock()

    @staticmethod
    def _build_embedding_client():
        if not settings.dashscope_api_key:
            logger.info("[IntentRouter] DASHSCOPE_API_KEY 未配置，Embedding 层已禁用")
            return None
        return AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    async def initialize(self) -> None:
        if self._embedding_disabled or self._intent_index is not None:
            return

        async with self._initialize_lock:
            if self._embedding_disabled or self._intent_index is not None:
                return
            try:
                self._intent_index = await IntentIndex.create(self._embedding_client)
                logger.info("[IntentRouter] Embedding 意图原型初始化完成")
            except Exception as exc:
                self._embedding_disabled = True
                logger.warning("[IntentRouter] Embedding 初始化失败，降级到 LLM: %s", exc)

    async def route(self, context: RouteContext) -> RouteDecision:
        if context.requested_action in {"create", "edit"}:
            return RouteDecision(
                intent=context.requested_action,
                source="explicit",
                confidence=1.0,
                reason=f"用户通过前端明确选择{context.requested_action} PPT",
            )

        await self.initialize()
        if self._intent_index is not None:
            try:
                top_intent, top_score, second_score = await self._intent_index.match(
                    context.user_message
                )
                margin = top_score - second_score
                if (
                    top_intent in {"chat", "create", "edit"}
                    and top_score >= settings.intent_embedding_min_score
                    and margin >= settings.intent_embedding_margin
                ):
                    return RouteDecision(
                        intent=top_intent,
                        source="embedding",
                        confidence=max(0.0, min(1.0, top_score)),
                        reason=(
                            f"Embedding 高置信命中，score={top_score:.3f}, "
                            f"margin={margin:.3f}"
                        ),
                    )
                logger.info(
                    "[IntentRouter] Embedding 低置信: top=%s score=%.3f margin=%.3f",
                    top_intent,
                    top_score,
                    margin,
                )
            except Exception as exc:
                logger.warning("[IntentRouter] Embedding 匹配失败，降级到 LLM: %s", exc)

        briefing = (
            f"当前用户消息：{context.user_message}\n"
            f"当前活动 PPT ID：{context.active_ppt_id or '无'}\n"
            f"当前风格：{context.style}\n"
            "请输出意图分类 JSON。"
        )
        try:
            result = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=INTENT_ROUTER_PROMPT),
                    *context.recent_messages[-6:],
                    HumanMessage(content=briefing),
                ]
            )
            return RouteDecision(
                intent=result.intent,
                source="llm",
                confidence=None,
                reason=result.reason,
            )
        except Exception as exc:
            logger.warning("[IntentRouter] LLM 分类失败，安全回退 chat: %s", exc)
            return RouteDecision(
                intent="chat",
                source="fallback",
                confidence=None,
                reason="意图识别服务不可用，安全回退普通对话",
            )
