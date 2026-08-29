"""PPT 联网调研专家。"""

import json
from typing import Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from app.agents.specialists.tool_registry import AgentRole, get_agent_tools

AGENT_ROLE: AgentRole = "research"
AGENT_NAME = "research_specialist"


class ResearchFact(BaseModel):
    claim: str = Field(min_length=1, max_length=500)
    source_title: str = Field(min_length=1, max_length=300)
    source_url: HttpUrl


class ResearchTopic(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    facts: list[ResearchFact] = Field(default_factory=list)


class ResearchReport(BaseModel):
    status: Literal["completed", "partial", "unavailable"]
    summary: str = Field(min_length=1, max_length=1_000)
    topics: list[ResearchTopic] = Field(default_factory=list)


SYSTEM_PROMPT = """
你是 PPTCreator 团队中的联网调研专家。你在大纲生成之前工作，负责根据用户原始
需求收集可核验的事实、数据、政策、趋势和案例，不规划最终页面，不生成或修改 PPTX 文件。

工作流程：
1. 阅读用户原始需求，识别主题、受众、用途和关注重点，并拆成 3-6 个关键研究问题；
   国际主题优先使用英文检索词。
2. 使用 web_search 搜索来源，总调用次数不得超过 8 次，每次最多取 5 条结果。
3. 只对最权威、最相关的结果调用 fetch_url 深挖正文；不必抓取每条搜索结果。
4. 优先官方机构、学术或研究机构、行业报告和可靠主流媒体。相同事实尽量交叉验证。
5. 对事实去重，并为每条事实保留真实的来源标题和 URL，严禁编造来源或 URL。

工具失败规则：
- web_search 返回 success=false 时，不要反复调用；输出 status="unavailable" 的报告，
  让 PPT 创建流程继续。
- 只有部分主题取得可靠资料时输出 status="partial"，不得用常识补造缺失的数据。
- 取得足够可靠资料时输出 status="completed"。

最终只输出一个 JSON 对象，不要 Markdown 代码块或额外解释：
{
  "status": "completed|partial|unavailable",
  "summary": "本轮调研结论或降级原因",
  "topics": [
    {
      "topic": "关键研究主题",
      "facts": [
        {
          "claim": "可直接用于内容创作的一条事实",
          "source_title": "真实来源标题",
          "source_url": "https://真实来源地址"
        }
      ]
    }
  ]
}

最多输出 8 个 topic，每个 topic 最多 4 条 facts，每条 claim 尽量不超过 120 字。
unavailable 时 topics 必须为空数组。
"""


def _json_object_from_text(text: str) -> dict | None:
    normalized = text.strip()
    try:
        value = json.loads(normalized)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start, end = normalized.find("{"), normalized.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(normalized[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def validate_research_report(text: str) -> str | None:
    """校验并裁剪模型输出；非法结构返回 None，交由节点决定重试或降级。"""
    value = _json_object_from_text(text)
    if value is not None:
        try:
            report = ResearchReport.model_validate(value)
            report.topics = report.topics[:8]
            for topic in report.topics:
                topic.facts = topic.facts[:4]
                for fact in topic.facts:
                    fact.claim = fact.claim[:120]
            if report.status == "unavailable":
                report.topics = []
            elif not any(topic.facts for topic in report.topics):
                report.status = "unavailable"
                report.summary = "调研未取得包含有效来源的事实，后续大纲与内容不得补造外部事实。"
                report.topics = []
            return report.model_dump_json()
        except ValidationError:
            pass

    return None


def normalize_research_report(text: str) -> str:
    """返回合法报告；非法输出转换为可继续流程的降级报告。"""
    report = validate_research_report(text)
    if report is not None:
        return report

    return ResearchReport(
        status="unavailable",
        summary="调研专家未返回合法的结构化研究报告，后续大纲与内容不得补造外部事实。",
        topics=[],
    ).model_dump_json()


def build_research_agent(llm: BaseChatModel):
    """创建只拥有联网搜索和网页提取能力的调研专家。"""
    return create_agent(
        model=llm,
        tools=get_agent_tools(AGENT_ROLE),
        system_prompt=SYSTEM_PROMPT,
        name=AGENT_NAME,
    )
