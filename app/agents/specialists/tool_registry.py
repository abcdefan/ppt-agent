"""专业子 Agent 的角色与业务工具注册表。"""

from typing import Literal

from langchain_core.tools import BaseTool

from app.tools import (
    add_chart_slide,
    add_image_slide,
    enhance_ppt,
    fetch_url,
    generate_ppt,
    prepare_chart_operation,
    prepare_image_operation,
    refine_content,
    web_search,
)

AgentRole = Literal[
    "outline",
    "research",
    "content",
    "image",
    "chart",
    "beautify",
]

AGENT_ROLES: tuple[AgentRole, ...] = (
    "outline",
    "research",
    "content",
    "image",
    "chart",
    "beautify",
)


_TOOLS_BY_ROLE: dict[AgentRole, list[BaseTool]] = {
    # 当前产品只支持根据用户主题直接创作，还没有上传参考文档的入口。
    # 因此大纲专家无需扫描 workspace；待支持上传后，再按请求显式授予
    # 对应文档的读取能力，避免误读历史文件或把 PPTX 当 UTF-8 文本读取。
    "outline": [],
    "research": [web_search, fetch_url],
    "content": [refine_content, generate_ppt],
    "image": [add_image_slide],
    "chart": [add_chart_slide],
    "beautify": [enhance_ppt],
}

_PREPARATION_TOOLS_BY_ROLE: dict[AgentRole, list[BaseTool]] = {
    **_TOOLS_BY_ROLE,
    "image": [prepare_image_operation],
    "chart": [prepare_chart_operation],
}


def get_agent_tools(
    agent_role: AgentRole,
    *,
    prepare_assets: bool = False,
) -> list[BaseTool]:
    """返回指定 Agent 的工具；准备模式不允许 Image/Chart 直接写 PPT。"""
    registry = _PREPARATION_TOOLS_BY_ROLE if prepare_assets else _TOOLS_BY_ROLE
    return list(registry[agent_role])
