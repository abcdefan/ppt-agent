"""将 Specialist Agents 包装成 Master Agent 可调用的委派 Tools。"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from app.agents.specialists.agent_registry import build_specialist_agent
from app.agents.specialists.tool_registry import AGENT_ROLES, AgentRole

_TOOL_DESCRIPTIONS: dict[AgentRole, str] = {
    "outline": (
        "调用大纲专家规划 PPT 页面结构。"
        "任务描述应包含主题、受众、页数、风格、补充要求和研究报告全文；"
        "返回包含 title、purpose 和 layout_hint 的大纲 JSON。"
    ),
    "research": (
        "调用联网调研专家根据用户原始需求收集可靠事实、最新数据、政策、趋势和案例。"
        "任务描述必须包含用户原始需求、主题、受众和用途；"
        "返回带 source_title/source_url 的结构化研究报告。"
    ),
    "content": (
        "调用内容专家生成幻灯片内容和 PPTX 文件。"
        "任务描述必须包含主题、风格、页数、大纲专家返回的完整 JSON，"
        "以及已执行的调研报告全文；调研不可用时必须明确说明；"
        "返回生成结果和真实 PPT 文件名。"
    ),
    "image": (
        "调用配图专家为已有 PPT 添加图片。"
        "任务描述必须包含真实 PPT 文件名、配图概念和风格要求。"
    ),
    "chart": (
        "调用图表专家为已有 PPT 添加数据图表。"
        "任务描述必须包含真实 PPT 文件名、图表类型和数据。"
    ),
    "beautify": (
        "调用美化专家优化已有 PPT 的布局和视觉效果。"
        "任务描述必须包含真实 PPT 文件名和风格要求。"
    ),
}


def _extract_final_text(result: dict) -> str:
    """从 Specialist Agent 运行结果中提取最后一条有效的 AI 文本。"""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = str(message.text).strip()
            if text:
                return text

    return "子 Agent 执行完成，但没有返回文字结果。"


def build_delegation_tool(
    agent_role: AgentRole,
    llm: BaseChatModel,
) -> BaseTool:
    """创建指定 Specialist Agent，并将其包装成 Master 的委派 Tool。

    内层 delegate_to_specialist() 会闭包引用本次创建的 specialist_agent。
    即使本函数执行结束，只要返回的 Tool 仍被 Master 保存，该 Specialist
    Agent 就不会被释放。例如 outline_agent_tool → delegate_to_specialist
    闭包 → outline_specialist。
    """
    specialist_agent = build_specialist_agent(
        agent_role=agent_role,
        llm=llm,
    )
    tool_name = f"{agent_role}_agent_tool"

    @tool(
        tool_name,
        description=_TOOL_DESCRIPTIONS[agent_role],
    )
    # 调用外层 build_delegation_tool() 时，执行到 async def 只会创建函数对象，
    # 不会执行下面的函数体；@tool 随即将该函数包装成名为 tool_name 的
    # BaseTool。等 Master 真正调用这个 Tool 并传入 task 时，函数体才会运行。
    async def delegate_to_specialist(task: str) -> str:
        """把 Master 生成的任务委派给 Specialist Agent 执行。"""
        # task 是 Master LLM 发起 Tool Call 时，根据用户需求、对话上下文和
        # 上游 Specialist Agent 返回结果重新组织出的委派任务。这里将它包装成
        # 一条新的 HumanMessage，相当于由 Master 代替用户向当前 Specialist
        # Agent 提问并启动其工作。Specialist Agent 不会自动继承 Master 的完整
        # 消息历史，因此它所需的大纲、文件名、风格等上下文，都必须由 Master
        # 明确写入 task 后才能传递进来。
        result = await specialist_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=task),
                ]
            }
        )
        return _extract_final_text(result)

    return delegate_to_specialist


def build_delegation_tools(
    llm: BaseChatModel,
) -> list[BaseTool]:
    """创建全部 Specialist Agents，并批量包装成 Master 委派 Tools。"""
    return [
        build_delegation_tool(
            agent_role=agent_role,
            llm=llm,
        )
        for agent_role in AGENT_ROLES
    ]
