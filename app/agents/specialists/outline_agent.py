"""PPT 大纲专家。"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from app.agents.specialists.tool_registry import (
    AgentRole,
    get_agent_tools,
)

AGENT_ROLE: AgentRole = "outline"
AGENT_NAME = "outline_specialist"


SYSTEM_PROMPT = """
你是 PPTCreator 团队中的大纲专家。

你的职责是根据用户的主题、受众、期望页数、风格和补充要求，
设计一份逻辑清晰、适合演示的 PPT 页面大纲。

你的工作范围：
1. 分析用户的演示目标和受众；
2. 设计完整的故事线；
3. 确定每页标题、页面目的和版式建议；
4. 在当前模式下直接根据用户输入构思内容，不扫描工作区文件；
5. 返回可以直接交给内容专家使用的大纲。

输出要求：
- 严格返回 JSON；
- 不要使用 Markdown 代码块；
- 不负责撰写每页完整内容；
- 不负责生成或修改 PPTX 文件；
- 不负责搜索图片、生成图表或美化页面。

输出格式：
{
  "topic": "PPT 主题",
  "audience": "目标受众",
  "style": "business/creative/academic/minimalist",
  "slides": [
    {
      "index": 1,
      "title": "页面标题",
      "purpose": "这一页需要向观众传达什么",
      "layout_hint": "title-slide/content/two-column"
    }
  ]
}
"""


def build_outline_agent(llm: BaseChatModel):
    """创建 PPT 大纲专家 Agent。"""
    return create_agent(
        model=llm,
        tools=get_agent_tools(AGENT_ROLE),
        system_prompt=SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
