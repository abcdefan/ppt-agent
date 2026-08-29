"""基于意图原型向量的轻量语义分类索引。

整体流程：

1. 为每个意图预先准备多条典型语句，例如 ``chat`` 和 ``create``。
2. 服务初始化时，使用同一个 Embedding 模型批量计算所有典型语句的向量。
3. 将同一意图下的语句向量取平均并归一化，得到该意图的原型向量。
   原型向量会缓存在当前 ``IntentIndex`` 实例中，不需要随每次请求重新计算。
4. 收到用户输入后，使用相同的 Embedding 模型计算查询向量并归一化。
5. 计算查询向量与每个意图原型向量的点积。由于两侧均已归一化，
   点积等价于余弦相似度。
6. 按相似度从高到低排序，返回 Top1 意图、Top1 分数和 Top2 分数。
7. ``IntentRouterService`` 再判断 Top1 是否达到最低相似度阈值，以及
   ``Top1 - Top2`` 是否达到最小 Margin：两项都满足才采用 Embedding 结果；
   否则交给结构化 LLM 继续分类。

这里的意图示例属于带标签的参考样例，并不会训练或微调 Embedding 模型。
本模块只负责生成原型和计算相似度，最终阈值决策由 Router Service 负责。
"""

import math

from app.core.config import settings

INTENT_EXAMPLES: dict[str, list[str]] = {
    "chat": [
        "你好",
        "你是谁",
        "谢谢你的帮助",
        "解释一下什么是多智能体",
        "这个概念是什么意思",
        "帮我回答一个普通问题",
        "今天天气怎么样",
        "我们继续聊聊刚才的话题",
    ],
    "create": [
        "帮我做一份PPT",
        "生成一份幻灯片",
        "制作一个关于人工智能的演示文稿",
        "我要创建PPT",
        "帮我做一份项目答辩课件",
        "生成一份季度工作总结幻灯片",
        "把这个主题整理成演示文稿",
        "做一套可以汇报的PPT页面",
    ],
    "edit": [
        "帮我修改刚才的PPT",
        "把当前PPT美化一下",
        "替换第三页的图片",
        "给这份幻灯片加一个图表",
        "调整已有演示文稿的视觉风格",
        "删除当前PPT里的配图",
        "修改这份PPT的图表",
        "优化刚做好的幻灯片排版",
    ],
}


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


async def _embed_batched(client, texts: list[str]) -> list[list[float]]:
    output: list[list[float]] = []
    batch_size = settings.dashscope_embedding_batch_size
    for index in range(0, len(texts), batch_size):
        response = await client.embeddings.create(
            model=settings.dashscope_embedding_model,
            input=texts[index : index + batch_size],
        )
        output.extend(item.embedding for item in response.data)
    return output


class IntentIndex:
    def __init__(self, client, prototypes: dict[str, list[float]]):
        self._client = client
        self._prototypes = prototypes

    @classmethod
    async def create(cls, client) -> "IntentIndex":
        texts: list[str] = []
        owners: list[str] = []
        for intent, examples in INTENT_EXAMPLES.items():
            texts.extend(examples)
            owners.extend([intent] * len(examples))

        vectors = await _embed_batched(client, texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding 返回数量与输入示例数量不一致")

        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for intent, vector in zip(owners, vectors, strict=True):
            accumulator = sums.setdefault(intent, [0.0] * len(vector))
            for position, value in enumerate(vector):
                accumulator[position] += value
            counts[intent] = counts.get(intent, 0) + 1

        prototypes = {
            intent: _normalize([value / counts[intent] for value in accumulator])
            for intent, accumulator in sums.items()
        }
        return cls(client=client, prototypes=prototypes)

    async def match(self, text: str) -> tuple[str | None, float, float]:
        if not self._prototypes:
            return None, 0.0, 0.0

        vector = _normalize((await _embed_batched(self._client, [text]))[0])
        scores = {
            intent: sum(left * right for left, right in zip(vector, prototype))
            for intent, prototype in self._prototypes.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        return top_intent, top_score, second_score
