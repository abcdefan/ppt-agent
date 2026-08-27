"""workflow 模式 Supervisor 节点 —— 读取 state，决策下一步交给哪个专家（或 FINISH）。

从原 agents.py 拆出；specialist 提示词（ROLE_PROMPTS）在 app.agents.common，
specialist agent 构造在 app.agents。
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.workflow.prompts import SUPERVISOR_PROMPT
from app.agents.workflow.state import AgentState

logger = logging.getLogger(__name__)

# 图节点名
SUPERVISOR = "supervisor"


class RouteDecision(BaseModel):
    """Supervisor 的路由输出（with_structured_output 强制结构化）"""

    next: Literal["outline", "research", "content", "image", "chart", "beautify", "FINISH"] = Field(
        description="下一步交给哪个专家，或 FINISH 表示全部完成"
    )
    reason: str = Field(default="", description="一句话决策理由")


# supervisor 合法的路由取值（json_mode 不传 schema，需代码兜底校验）
VALID_NEXT = ("outline", "research", "content", "image", "chart", "beautify", "FINISH")


def build_supervisor_node(llm):
    """构建 supervisor 节点：读取 state，决策 next。

    method="json_mode"：DashScope(qwen3.5 thinking) 模型下，function_calling 方式
    会因 'tool_choice does not support required in thinking mode' 报错；json_mode
    可行，但要求 messages 含 'json' 字样（已在 SUPERVISOR_PROMPT 输出格式段满足），
    并对解析结果做合法性兜底（json_mode 不强制 schema）。
    """
    structured = llm.with_structured_output(RouteDecision, method="json_mode")

    async def node(state: AgentState) -> dict:
        filename = state.get("filename") or "（尚未生成）"
        outline = "已有大纲" if state.get("outline") else "尚未生成大纲"
        research = "已有研究笔记" if state.get("research") else "尚未生成研究笔记"
        style = state.get("style") or "business"
        briefing = (
            f"用户需求：{state.get('user_message', '')}\n"
            f"大纲：{outline}\n"
            f"研究笔记：{research}\n"
            f"当前 PPT 文件：{filename}\n"
            f"主题风格：{style}\n"
            "请决定下一步交给哪个专家（outline/research/content/image/chart/beautify），或 FINISH。"
        )
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            *state["messages"][-6:],   # 近期上下文，避免 prompt 过长
            HumanMessage(content=briefing),
        ]
        next_val = "FINISH"
        reason = ""
        try:
            decision: RouteDecision = await structured.ainvoke(messages)
            if decision and getattr(decision, "next", None) in VALID_NEXT:
                next_val = decision.next
                reason = decision.reason or ""
            else:
                logger.warning("[supervisor] 结构化输出非法，回退 FINISH: %r", decision)
        except Exception as e:
            # 解析失败不能让整个图崩溃：默认结束，避免死循环
            logger.warning("[supervisor] 结构化输出异常，回退 FINISH: %s", e)

        logger.info("[supervisor] next=%s reason=%s", next_val, reason)
        return {
            "next": next_val,
            "messages": [
                HumanMessage(content=f"[协调员 → {next_val}] {reason}")
            ],
        }

    return node
