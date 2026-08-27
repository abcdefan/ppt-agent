"""Workflow Supervisor Node。

Supervisor 不把 Specialist Agents 当作 Tools 调用。它只读取当前 WorkflowState，
让 LLM 生成结构化路由决定，再把 ``next`` 写回 State。Graph 根据 ``next`` 的值
通过条件边进入对应 Specialist Node 或结束流程。
"""

import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.workflow.state import WorkflowState

# Supervisor 在 Graph 中注册时使用的节点名称。
SUPERVISOR = "supervisor"


SUPERVISOR_PROMPT = """你是 PPT 多智能体团队的 Supervisor，只负责根据当前
工作流状态决定下一步应该执行哪个 Specialist Agent，不直接生成或修改 PPT。

可选择的节点：
- outline：规划 PPT 页面大纲；
- research：根据大纲联网检索最新事实、数据、政策、趋势和案例；
- content：根据大纲生成内容和基础 PPTX 文件；
- image：为已有 PPT 添加配图；
- chart：为已有 PPT 添加数据图表；
- beautify：对已有 PPT 进行最终视觉美化；
- FINISH：所有需要的工作已经完成，结束流程。

路由规则：
1. 没有大纲时选择 outline；
2. 大纲完成后，如果主题涉及最新数据、市场、趋势、政策、竞品、人物事件、
   外部案例、可核验事实，或用户明确要求来源，并且 research 尚未执行，则选择 research；
3. 纯创意、个人材料改写、模板排版或故事文案可跳过 research；边界不确定时选择 research；
4. research 已执行（包括 unavailable）后不得重试；报告不可用时让 content 跳过外部事实，
   严禁要求它猜测数据或来源；
5. 已有大纲且无需调研，或调研已经执行，但没有 PPT 文件时选择 content；
6. 已有 PPT 文件后，根据用户原始需求按需选择 image、chart 或 beautify；
7. completed_agents 中已经完成的专家不要重复调用；所有必要工作完成后选择 FINISH。

必须返回 JSON 对象，格式为：
{"next": "outline/research/content/image/chart/beautify/FINISH", "reason": "一句话理由"}
"""


class RouteDecision(BaseModel):
    """Supervisor LLM 的结构化路由结果。"""

    next: Literal[
        "outline",
        "research",
        "content",
        "image",
        "chart",
        "beautify",
        "FINISH",
    ] = Field(description="下一步执行的节点，或 FINISH")
    reason: str = Field(description="选择该路由的一句话理由")


def _safe_recent_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """只保留普通对话消息，阻止 Tool 协议消息进入 Supervisor。"""
    safe_messages = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]
    return safe_messages[-8:]


def build_supervisor_node(llm: BaseChatModel):
    """使用已有 LLM 创建 Supervisor Node 函数。"""
    # Supervisor 每次只需要做一次路由判断，不需要 Tools 和 ReAct 循环，因此
    # 不使用 create_agent()，而是直接要求 ChatModel 返回 RouteDecision。
    structured_llm = llm.with_structured_output(
        RouteDecision,
        method="json_mode",
    )

    async def run_supervisor(state: WorkflowState) -> dict:
        """读取当前 State，返回 next 字段的局部更新。"""
        outline_status = "已有大纲" if state.get("outline") else "尚无大纲"
        research_report = state.get("research_report")
        research_status = "尚未执行"
        if research_report:
            try:
                research_status = json.loads(research_report).get("status", "已执行")
            except (json.JSONDecodeError, TypeError, AttributeError):
                research_status = "已执行但报告格式未知"
        filename = state.get("filename") or "尚未生成 PPT 文件"
        completed_agents = state.get("completed_agents", [])
        completed_status = "、".join(completed_agents) or "暂无"

        state_summary = (
            f"用户原始需求：{state['user_message']}\n"
            f"PPT 风格：{state['style']}\n"
            f"当前大纲状态：{outline_status}\n"
            f"当前调研状态：{research_status}\n"
            f"当前 PPT 文件：{filename}\n"
            f"已完成的专家：{completed_status}\n"
            "请根据结构化状态决定下一步，不要重复调用已完成的专家。"
        )

        decision = await structured_llm.ainvoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                # Specialist 的 Tool Call/Tool Result 属于其私有协议上下文，
                # Supervisor 只接收普通消息。即使未来有节点误写 ToolMessage，
                # 这里也会进行最后一道防御性过滤。
                *_safe_recent_messages(state["messages"]),
                HumanMessage(content=state_summary),
            ]
        )

        # next 供 Graph 的条件边读取；路由理由作为内部消息留在
        # State 中，便于 Supervisor 下一次判断。这里不使用
        # AIMessage，避免 Runner 在结束时把“路由到 FINISH”误当成
        # Specialist 给用户的最终回复。
        return {
            "next": decision.next,
            "messages": [
                HumanMessage(
                    content=f"路由到 {decision.next}：{decision.reason}",
                    name=SUPERVISOR,
                )
            ],
        }

    return run_supervisor
