"""大纲专家"""

from langchain.agents import create_agent

from app.agents.common.prompts import ROLE_PROMPTS
from app.agents.common.tools import split_tools

ROLE = "outline"


def build_outline_specialist(llm):
    """返回 outline 专家的 agent（输出幻灯片结构 JSON，read_file/list_files 供参考）。"""
    return create_agent(
        model=llm,
        tools=split_tools()[ROLE],
        system_prompt=ROLE_PROMPTS[ROLE],
    )
