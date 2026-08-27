"""美化专家 specialist —— 对已有 PPT 做布局/装饰/视觉美化。

封装该角色的 create_agent；供 workflow（图节点）和 subagents（@tool）两种模式复用。
提示词见 common/prompts.py:BEAUTIFY_PROMPT，工具子集见 common/tools.py。
"""

from langchain.agents import create_agent

from app.agents.common.prompts import ROLE_PROMPTS
from app.agents.common.tools import split_tools

ROLE = "beautify"


def build_beautify_specialist(llm):
    """返回 beautify 专家的 agent（enhance_ppt → read_file）。"""
    return create_agent(
        model=llm,
        tools=split_tools()[ROLE],
        system_prompt=ROLE_PROMPTS[ROLE],
    )
