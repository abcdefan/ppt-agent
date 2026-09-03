"""应用可观测性初始化。"""

import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


def setup_observability() -> None:
    """将应用配置桥接为 LangSmith SDK 识别的进程环境变量。

    Pydantic Settings 读取 ``.env`` 后只会把值保存在 ``settings`` 对象中，
    不会自动写入 ``os.environ``。LangChain/LangGraph 的 LangSmith 集成读取
    的是后者，因此必须在创建和执行任何 Agent/Graph 之前完成这一步。
    """
    tracing_enabled = settings.langsmith_tracing
    os.environ["LANGSMITH_TRACING"] = (
        "true" if tracing_enabled else "false"
    )

    if not tracing_enabled:
        logger.info("LangSmith tracing 已关闭")
        return

    api_key = settings.langsmith_api_key.strip()
    if not api_key:
        raise RuntimeError("LANGSMITH_TRACING=true 时必须配置 LANGSMITH_API_KEY")

    os.environ["LANGSMITH_API_KEY"] = api_key

    project = settings.langsmith_project.strip()
    if project:
        os.environ["LANGSMITH_PROJECT"] = project

    endpoint = settings.langsmith_endpoint.strip()
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    # 不记录 API Key，只输出可以安全用于排查的启用状态和目标位置。
    logger.info(
        "LangSmith tracing 已启用: project=%s endpoint=%s",
        project or "default",
        endpoint or "default",
    )
