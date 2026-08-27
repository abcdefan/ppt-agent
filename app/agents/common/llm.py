import logging

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_llm(
    *,
    api_key: str = settings.deepseek_api_key,
    base_url: str = settings.deepseek_base_url,
    model: str = settings.deepseek_model,
    temperature: float = 0.6,
    streaming: bool = True,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        streaming=streaming,
    )


def build_summary_llm() -> ChatOpenAI:
    """创建用于会话摘要的低温、非流式模型。"""
    return build_llm(
        temperature=0.0,
        streaming=False,
    )
