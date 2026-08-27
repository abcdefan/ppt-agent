"""内容专家 specialist —— 拿大纲填充内容并生成 PPT。

封装该角色的 create_agent；供 workflow（图节点）和 subagents（@tool）两种模式复用。
提示词见 common/prompts.py:CONTENT_PROMPT，工具子集见 common/tools.py。
"""

from langchain.agents import create_agent

from app.agents.common.prompts import ROLE_PROMPTS
from app.agents.common.tools import split_tools

ROLE = "content"


def build_content_specialist(llm):
    """返回 content 专家的 agent（refine_content → generate_ppt → read_file/list_files）。"""
    return create_agent(
        model=llm,
        tools=split_tools()[ROLE],
        system_prompt=ROLE_PROMPTS[ROLE],
    )
