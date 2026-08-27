"""app/agents —— 5 个 specialist 各自一个文件，供 workflow / subagents 两种模式复用。

每个 specialist 封装一次 create_agent；workflow 包成图节点、subagents 包成 @tool，
都从这里取底层 agent。提示词（ROLE_PROMPTS）与工具分组（split_tools）见 app.agents.common。

推荐流水线：outline → content（生成 PPTX）→ image → chart → beautify。
"""

from app.agents.beautify_agent import build_beautify_specialist
from app.agents.chart_agent import build_chart_specialist
from app.agents.content_agent import build_content_specialist
from app.agents.image_agent import build_image_specialist
from app.agents.outline_agent import build_outline_specialist
from app.agents.research_agent import build_research_specialist

# 角色名 → 构造器注册表，供两种模式按 ROLES 统一遍历
SPECIALIST_BUILDERS = {
    "outline": build_outline_specialist,
    "research": build_research_specialist,
    "content": build_content_specialist,
    "image": build_image_specialist,
    "chart": build_chart_specialist,
    "beautify": build_beautify_specialist,
}


def build_specialist(role: str, llm):
    """按角色名构造 specialist ReAct agent。"""
    try:
        return SPECIALIST_BUILDERS[role](llm)
    except KeyError:
        raise ValueError(f"未知 specialist 角色: {role!r}，可选: {sorted(SPECIALIST_BUILDERS)}")


__all__ = [
    "build_outline_specialist",
    "build_research_specialist",
    "build_content_specialist",
    "build_image_specialist",
    "build_chart_specialist",
    "build_beautify_specialist",
    "SPECIALIST_BUILDERS",
    "build_specialist",
]
