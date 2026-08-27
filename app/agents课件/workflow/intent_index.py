"""意图向量索引 — embedding 预筛用。

为 workflow 的 intent_router 提供"语义相似度"快速分类：预计算四类意图示例的
归一化平均向量（原型），运行时把用户消息 embed 后按余弦相似度取 top1。
router 结合 min_score / margin 决定直接采用或走 LLM 兜底。

注意：
- 用 openai.AsyncOpenAI 直连 DashScope（绕 langchain OpenAIEmbeddings 的 tiktoken 预分词坑）。
- DashScope text-embedding-v3 单次批量上限 10，_embed_batched 自动分批。
- 效果依赖 INTENT_EXAMPLES 的覆盖度，可按需扩充。
"""

import logging
import math
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# DashScope text-embedding-v3 单次批量上限
EMBEDDING_BATCH = 10

# 四类意图示例（典型表述；扩充示例可提升召回与边界区分度）
INTENT_EXAMPLES: dict[str, list[str]] = {
    "chat": [
        "你好呀",
        "今天天气怎么样",
        "你是谁",
        "谢谢啦",
        "在吗",
        "帮我查个常识问题",
    ],
    "create": [
        "帮我做一份PPT",
        "生成一份幻灯片",
        "做一个关于AI的演示文稿",
        "我要创建PPT",
        "给我做一份汇报幻灯片",
        "做一份产品介绍PPT",
    ],
    "enhance": [
        "把这份PPT美化一下",
        "再给PPT加个图表",
        "换一下幻灯片的配图",
        "优化一下排版",
        "美化已有的PPT",
        "给这份演示文稿加配图",
    ],
    "analyze": [
        "这份PPT讲了什么",
        "帮我看看这个演示文稿",
        "总结一下内容",
        "分析一下结构",
        "这份演示文稿的主旨",
        "概括一下这份PPT",
    ],
}


async def _embed_batched(client, texts: list[str]) -> list[list[float]]:
    """分批 embed（DashScope 单次≤10）。"""
    out: list[list[float]] = []
    model = settings.dashscope_embedding_model
    for i in range(0, len(texts), EMBEDDING_BATCH):
        resp = await client.embeddings.create(
            model=model, input=texts[i:i + EMBEDDING_BATCH]
        )
        out.extend(d.embedding for d in resp.data)
    return out


def _normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class IntentIndex:
    """意图向量索引：每类意图一个归一化原型向量，余弦相似度匹配。"""

    def __init__(self, client, protos: dict[str, list[float]]):
        self._client = client
        self._protos = protos  # 已归一化 {intent: unit_vec}

    @classmethod
    async def create(cls, client) -> "IntentIndex":
        """预计算所有示例的 embedding，按意图聚合为归一化平均向量。"""
        texts: list[str] = []
        owners: list[str] = []
        for intent, exs in INTENT_EXAMPLES.items():
            for t in exs:
                texts.append(t)
                owners.append(intent)

        vecs = await _embed_batched(client, texts)

        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for owner, v in zip(owners, vecs):
            acc = sums.setdefault(owner, [0.0] * len(v))
            for i, x in enumerate(v):
                acc[i] += x
            counts[owner] = counts.get(owner, 0) + 1

        protos = {
            intent: _normalize([x / counts[intent] for x in acc])
            for intent, acc in sums.items()
        }
        logger.info("[intent_index] 预计算完成：%d 类意图", len(protos))
        return cls(client, protos)

    async def match(self, text: str) -> tuple[Optional[str], float, float]:
        """返回 (top1_intent, top1_score, top2_score)；无原型时 (None, 0.0, 0.0)。"""
        if not self._protos:
            return None, 0.0, 0.0
        v = _normalize((await _embed_batched(self._client, [text]))[0])
        # 原型已归一化 → 余弦相似度 = 点积
        scores = {
            intent: sum(x * y for x, y in zip(v, p))
            for intent, p in self._protos.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top1_intent, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        return top1_intent, top1_score, top2_score
