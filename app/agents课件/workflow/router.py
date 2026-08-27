"""workflow 模式 Intent Router 节点 —— 入口意图分类 + chat 直答短路。

四类意图：
  chat    闲聊/简单问答（不走 specialist，直接 LLM 回复后 END）
  create  新建 PPT
  enhance 对已有 PPT 二次加工（加图/图表/美化）
  analyze 查询/分析（本轮无专门专家）

create/enhance/analyze 本轮一律交给现有 supervisor 串行循环；router 仅负责
(a) chat 短路避免在闲聊上触发 PPT 生成；(b) 把 intent 标签写入 state，供未来按意图跳过专家。
"""

import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.workflow.prompts import CHAT_REPLY_PROMPT, INTENT_ROUTER_PROMPT
from app.agents.workflow.intent_index import IntentIndex
from app.agents.workflow.state import AgentState
from app.agents.common.llm import build_embedding_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# 图节点名
INTENT_ROUTER = "intent_router"
CHAT_REPLY = "chat_reply"

# 四类合法意图（json_mode 不强制 Literal，需代码兜底）
VALID_INTENTS = ("chat", "create", "enhance", "analyze")


class IntentDecision(BaseModel):
    """Intent Router 的分类输出（with_structured_output 强制结构化）"""

    intent: Literal["chat", "create", "enhance", "analyze"] = Field(
        description="用户意图：chat/create/enhance/analyze 之一"
    )
    reason: str = Field(default="", description="一句话分类理由")


def build_intent_router_node(llm):
    """构建 intent_router 节点：embedding 语义预筛 + LLM 结构化兜底。

    流程：
    1) embedding 预筛：用户消息与四类意图原型算余弦，若 top1 ≥ min_score 且
       (top1 - top2) ≥ margin → 直接采用（快，省一次 LLM 调用）。
    2) 否则走 LLM 兜底：with_structured_output(IntentDecision, method="json_mode")。
       json_mode 与 supervisor 同因（thinking 模型下 function_calling 报错），
       prompt 已含 'json' 字样；结果做合法性兜底。
    3) 任何异常/非法输出 → 降级 chat（最保守：不创建文件、不扣配额，转 chat_reply 对话直答）。

    IntentIndex 在首次调用时 lazy 预计算（async），之后复用；构建失败则全程走 LLM。
    """
    structured = llm.with_structured_output(IntentDecision, method="json_mode")
    emb_client = build_embedding_client()
    cache: dict = {"index": None}  # None=未构建 / False=不可用 / IntentIndex=就绪

    async def _get_index():
        if cache["index"] is None:
            try:
                cache["index"] = await IntentIndex.create(emb_client)
            except Exception as e:
                logger.warning("[intent_router] 向量索引构建失败，全程走 LLM 兜底: %s", e)
                cache["index"] = False
        return cache["index"] if cache["index"] is not False else None

    async def _llm_classify(state: AgentState) -> str:
        user_message = state.get("user_message", "")
        has_ppt = "已有 PPT" if state.get("filename") else "尚无 PPT"
        briefing = (
            f"用户消息：{user_message}\n"
            f"当前状态：{has_ppt}\n"
            "请输出意图分类 JSON。"
        )
        messages = [
            SystemMessage(content=INTENT_ROUTER_PROMPT),
            *state["messages"][-4:],   # 仅近期上下文，分类无需长历史
            HumanMessage(content=briefing),
        ]
        try:
            decision: IntentDecision = await structured.ainvoke(messages)
            if decision and getattr(decision, "intent", None) in VALID_INTENTS:
                logger.info("[intent_router] LLM 兜底: intent=%s", decision.intent)
                return decision.intent
            logger.warning("[intent_router] LLM 输出非法，回退 chat: %r", decision)
        except Exception as e:
            logger.warning("[intent_router] LLM 兜底异常，回退 chat: %s", e)
        return "chat"

    async def node(state: AgentState) -> dict:
        user_message = state.get("user_message", "")
        intent_val: Optional[str] = None
        route = "llm"  # 默认走 LLM；embedding 命中时改写为 "embedding"

        # 1. embedding 预筛
        index = await _get_index()
        if index is not None:
            try:
                top1, s1, s2 = await index.match(user_message)
                margin = s1 - s2
                if (
                    top1
                    and s1 >= settings.intent_embedding_min_score
                    and margin >= settings.intent_embedding_margin
                ):
                    intent_val = top1
                    route = "embedding"
                    logger.info(
                        "[intent_router] embedding 命中: %s (s=%.3f margin=%.3f)",
                        top1, s1, margin,
                    )
                else:
                    logger.info(
                        "[intent_router] embedding 低置信 (top=%s s=%.3f margin=%.3f)，走 LLM",
                        top1, s1, margin,
                    )
            except Exception as e:
                logger.warning("[intent_router] embedding 匹配异常，走 LLM: %s", e)

        # 2. LLM 兜底（失败降级 create）
        if intent_val is None:
            intent_val = await _llm_classify(state)

        # 统一打印最终意图（无论走 embedding 还是 LLM，都会输出这一行）
        logger.info(
            "[intent_router] >>> 最终意图: %s （路径: %s） | 消息: %s",
            intent_val, route, user_message[:60],
        )
        return {"intent": intent_val}

    return node


def build_chat_reply_node(llm):
    """构建 chat_reply 节点：直接对话回复（不走 specialist），完成后 END。

    用 plain llm（不 bind_tools）：闲聊/简单问答无需工具；若误绑工具，
    DashScope 可能在闲聊上幻觉调用 generate_ppt，违背 chat 短路初衷。
    用 astream 产出，token 会被 streaming.py 的 on_chat_model_stream 捕获并推送。
    """

    async def node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=CHAT_REPLY_PROMPT),
            *state["messages"][-8:],
        ]
        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk)
        full = "".join(getattr(c, "content", "") or "" for c in chunks)
        return {"messages": [AIMessage(content=full)]}

    return node
