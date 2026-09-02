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

INTENT_ROUTER_PROMPT = """你是 PPTCreator 的入口请求路由器。

先判断用户讨论的语义意图，只允许以下三种：
- chat：问候、闲聊、知识问答、解释概念或普通对话，不要求生成 PPT 文件。
- create：用户要求制作、创建、生成演示文稿、幻灯片或 PPT 文件。
- edit：用户要求修改、替换、美化、增删已有 PPT 的图片、图表或视觉样式。

再独立判断 execute，表示用户是否明确要求本轮立即执行 Create/Edit 文件操作：
- 完整、直接、可立即操作的创建/修改命令，execute=true；不强制要求固定口令。
- 当前消息明确要求“开始做、开始修改、按上面要求执行”等，结合近期上下文
  可以确定此前收集的是 Create 或 Edit 时，execute=true。
- 用户只表达计划、讨论方案、准备继续描述、要求先听后续诉求，execute=false。
- 用户说“先不要做、等我说完、还有要求”等，execute=false，即使历史中曾要求执行。
- 信息不足或不确定是否应该开始时，execute=false。
- intent=chat 时 execute 必须为 false。

近期上下文用于理解“上面的要求”和正在讨论的任务；是否执行以当前用户消息为主，
不能仅凭历史中的执行措辞启动本轮操作。不要因为系统是 PPTCreator 就把普通问题
判为 create，也不要因为出现“创建、修改”等词就自动令 execute=true。

如果输入中提供了“已确定语义意图”，必须保持该 intent，只判断 execute。
必须返回 JSON：
{"intent":"chat/create/edit","execute":true/false,"reason":"一句话理由"}。
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
                logger.warning(
                    "[IntentRouter] Embedding 初始化失败，降级到 LLM: %s", exc
                )

    async def route(self, context: RouteContext) -> RouteDecision:
        intent_hint = None
        intent_source = "llm"
        intent_confidence = None
        intent_reason = ""

        if context.requested_action in {"create", "edit"}:
            intent_hint = context.requested_action
            intent_source = "explicit"
            intent_confidence = 1.0
            intent_reason = f"前端明确指定 {context.requested_action} 意图"
        else:
            await self.initialize()

        if intent_hint is None and self._intent_index is not None:
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
                    intent_hint = top_intent
                    intent_source = "embedding"
                    intent_confidence = max(0.0, min(1.0, top_score))
                    intent_reason = (
                        f"Embedding 高置信命中，score={top_score:.3f}, "
                        f"margin={margin:.3f}"
                    )
                else:
                    logger.info(
                        "[IntentRouter] Embedding 低置信: top=%s score=%.3f margin=%.3f",
                        top_intent,
                        top_score,
                        margin,
                    )
            except Exception as exc:
                logger.warning("[IntentRouter] Embedding 匹配失败，降级到 LLM: %s", exc)

        intent_hint_text = intent_hint or "无，由你判断"
        briefing = (
            f"当前用户消息：{context.user_message}\n"
            f"当前活动 PPT ID：{context.active_ppt_id or '无'}\n"
            f"当前风格：{context.style}\n"
            f"已确定语义意图：{intent_hint_text}\n"
            "请输出意图与是否立即执行的 JSON。"
        )
        try:
            result = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=INTENT_ROUTER_PROMPT),
                    *context.recent_messages[-6:],
                    HumanMessage(content=briefing),
                ]
            )
            final_intent = intent_hint or result.intent
            execute = bool(result.execute) if final_intent != "chat" else False
            reason = result.reason
            if intent_reason:
                reason = f"{intent_reason}；{reason}"
            return RouteDecision(
                intent=final_intent,
                execute=execute,
                source=intent_source,
                confidence=intent_confidence,
                reason=reason,
            )
        except Exception as exc:
            logger.warning("[IntentRouter] LLM 路由失败，安全禁止执行: %s", exc)
            return RouteDecision(
                intent=intent_hint or "chat",
                execute=False,
                source=intent_source if intent_hint else "fallback",
                confidence=intent_confidence,
                reason=(
                    f"{intent_reason}；执行判断不可用，安全禁止执行"
                    if intent_reason
                    else "意图与执行判断不可用，安全回退普通对话"
                ),
            )
