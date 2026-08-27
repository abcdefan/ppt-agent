"""配图专家 specialist —— 为已有 PPT 追加图片页。

封装该角色的 create_agent；供 workflow（图节点）和 subagents（@tool）两种模式复用。
提示词见 common/prompts.py:IMAGE_PROMPT，工具子集见 common/tools.py。
"""

from langchain.agents import create_agent

from app.agents.common.prompts import ROLE_PROMPTS
from app.agents.common.tools import split_tools

ROLE = "image"


def build_image_specialist(llm):
    """返回 image 专家的 agent（add_image_slide → read_file）。"""
    return create_agent(
        model=llm,
        tools=split_tools()[ROLE],
        system_prompt=ROLE_PROMPTS[ROLE],
    )
