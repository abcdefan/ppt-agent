"""调研专家（research）specialist —— 读大纲联网检索，产出带引用的研究笔记 JSON。

封装该角色的 create_agent；供 workflow（图节点）和 subagents（@tool）两种模式复用。
提示词见 common/prompts.py:RESEARCH_PROMPT，工具子集见 common/tools.py。
"""

from langchain.agents import create_agent

from app.agents.common.prompts import ROLE_PROMPTS
from app.agents.common.tools import split_tools

ROLE = "research"


def build_research_specialist(llm):
    """返回 research 专家的 agent（web_search / fetch_page / read_file / list_files）。"""
    return create_agent(
        model=llm,
        tools=split_tools()[ROLE],
        system_prompt=ROLE_PROMPTS[ROLE],
    )
