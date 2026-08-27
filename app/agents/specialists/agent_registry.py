"""可供不同编排方式复用的 Specialist Agent 注册表。"""

import logging
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from app.agents.specialists.beautify_agent import build_beautify_agent
from app.agents.specialists.chart_agent import build_chart_agent
from app.agents.specialists.content_agent import build_content_agent
from app.agents.specialists.image_agent import build_image_agent
from app.agents.specialists.outline_agent import build_outline_agent
from app.agents.specialists.research_agent import build_research_agent
from app.agents.specialists.tool_registry import AgentRole

logger = logging.getLogger(__name__)

# Agent 构造函数类型：接收一个聊天模型，返回一个创建好的 Agent 对象。
AgentBuilder = Callable[[BaseChatModel], object]


_SPECIALIST_AGENT_BUILDERS: dict[AgentRole, AgentBuilder] = {
    "outline": build_outline_agent,
    "research": build_research_agent,
    "content": build_content_agent,
    "image": build_image_agent,
    "chart": build_chart_agent,
    "beautify": build_beautify_agent,
}


def build_specialist_agent(
    agent_role: AgentRole,
    llm: BaseChatModel,
    *,
    prepare_assets: bool = False,
):
    """根据角色创建专业 Agent，供不同的多 Agent 编排方式复用。"""
    try:
        builder = _SPECIALIST_AGENT_BUILDERS[agent_role]
    except KeyError as exc:
        available_roles = ", ".join(_SPECIALIST_AGENT_BUILDERS)
        logger.warning(
            "请求创建未注册的 Specialist Agent: role=%s, available=%s",
            agent_role,
            available_roles,
        )
        raise ValueError(
            f"尚未注册角色为 {agent_role!r} 的 Specialist Agent；"
            f"当前可用角色: {available_roles}"
        ) from exc

    try:
        if agent_role == "image":
            specialist_agent = build_image_agent(
                llm,
                prepare_assets=prepare_assets,
            )
        elif agent_role == "chart":
            specialist_agent = build_chart_agent(
                llm,
                prepare_assets=prepare_assets,
            )
        else:
            specialist_agent = builder(llm)
    except Exception:
        logger.exception("创建 Specialist Agent 失败: role=%s", agent_role)
        raise

    logger.info("Specialist Agent 创建完成: role=%s", agent_role)
    return specialist_agent
