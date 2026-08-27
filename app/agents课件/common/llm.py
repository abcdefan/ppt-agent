"""LLM 工厂 — 统一创建 DashScope(通义千问) ChatOpenAI 实例，供两种多智能体模式共享。"""

import logging

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# DashScope 兼容 OpenAI 的 API 地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_llm(temperature: float = 0.6) -> ChatOpenAI:
    """创建主 LLM。

    streaming=True 是关键：多智能体全程流式依赖 astream_events(version="v2")
    拿到 on_chat_model_stream 的 token 流，否则 TEXT_DELTA 无内容。
    """
    return ChatOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=DASHSCOPE_BASE_URL,
        model=settings.dashscope_model,
        temperature=temperature,
        streaming=True,
    )


def build_summary_llm() -> ChatOpenAI:
    """创建摘要 LLM（低温度，供 SummaryBufferMemory 使用）。"""
    return ChatOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=DASHSCOPE_BASE_URL,
        model=settings.dashscope_model,
        temperature=0.2,
    )


def build_embedding_client():
    """创建 DashScope embedding 客户端"""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=DASHSCOPE_BASE_URL,
    )
