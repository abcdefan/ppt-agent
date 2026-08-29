"""Chat/Create/Edit 三条 Workflow 分支共用的用户回复节点。"""

from langchain_core.messages import AIMessage

from app.agents.router import ChatResponder
from app.agents.workflow.state import WorkflowState

REPLY_NODE = "reply_node"
CREATE_REPLY_MARKER = "workflow_create_completion"

_AGENT_LABELS = {
    "outline": "大纲规划",
    "research": "联网调研",
    "content": "内容生成",
    "image": "配图处理",
    "chart": "图表生成",
    "writer": "资源统一写入",
    "beautify": "视觉美化",
}


def build_reply_node(chat_responder: ChatResponder):
    async def reply(state: WorkflowState) -> dict:
        if state.get("intent") in {"create", "edit"}:
            if state.get("ppt_context_error"):
                action_label = "创建" if state.get("intent") == "create" else "编辑"
                content = f"无法{action_label} PPT：{state['ppt_context_error']}"
                return {
                    "messages": [
                        AIMessage(
                            content=content,
                            additional_kwargs={
                                "workflow_reply_kind": CREATE_REPLY_MARKER,
                            },
                        )
                    ]
                }

            completed_agents = state.get("completed_agents", [])
            completed_text = "、".join(
                _AGENT_LABELS[agent]
                for agent in completed_agents
                if agent in _AGENT_LABELS
            )

            if state.get("workflow_error"):
                action_label = "创建" if state.get("intent") == "create" else "编辑"
                content = f"PPT {action_label}未完成：{state['workflow_error']}"
                if completed_text:
                    content += f"（已完成：{completed_text}）"
                content += "。"
                return {
                    "messages": [
                        AIMessage(
                            content=content,
                            additional_kwargs={
                                "workflow_reply_kind": CREATE_REPLY_MARKER,
                            },
                        )
                    ]
                }

            filename = state.get("filename")
            if filename and state.get("intent") == "edit":
                content = f"PPT 已修改完成，文件名为：**{filename}**。"
                if completed_text:
                    content += f"本次已完成{completed_text}。"
            elif filename:
                content = f"PPT 已制作完成，文件名为：**{filename}**。"
                if completed_text:
                    content += f"本次已完成{completed_text}。"
            else:
                content = "本轮 PPT 流程已结束，但没有检测到可用的 PPT 文件。"

            # Create 回复不调用 LLM，没有 on_chat_model_stream 事件；这个标记
            # 让 streaming.py 在 Node 结束时把确定性文案发送给前端。
            return {
                "messages": [
                    AIMessage(
                        content=content,
                        additional_kwargs={
                            "workflow_reply_kind": CREATE_REPLY_MARKER,
                        },
                    )
                ]
            }

        # Chat 回复由无 Tools 的 ChatResponder 流式生成。
        chunks = [
            chunk
            async for chunk in chat_responder.stream(
                user_message=state["user_message"],
                # 去掉 Workflow 自己的 Overview SystemMessage 和当前用户消息；
                # ChatResponder 会添加自己的 System Prompt 与当前输入。
                history=state.get("messages", [])[1:-1],
            )
        ]
        return {"messages": [AIMessage(content="".join(chunks))]}

    return reply
