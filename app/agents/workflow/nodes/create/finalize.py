"""Create 的统一业务校验与收尾节点。"""

from app.agents.workflow.state import WorkflowState

FINALIZE_CREATE_NODE = "finalize_create_node"


async def finalize_create_node(state: WorkflowState) -> dict:
    """校验 Create 的最小业务产物，并为持久化设置终态。"""
    missing: list[str] = []
    if not state.get("research_report"):
        missing.append("调研报告")
    if not state.get("outline"):
        missing.append("页面大纲")
    if not state.get("filename"):
        missing.append("PPT 文件")
    if not state.get("slides_manifest"):
        missing.append("页面内容清单")

    required = set(state.get("required_stages", []))
    completed = set(state.get("completed_stages", []))
    if "assets" in required and "assets" not in completed:
        missing.append("图片/图表资源处理")
    if "beautify" in required and "beautify" not in completed:
        missing.append("视觉美化")

    workflow_error = state.get("workflow_error")
    # 重试达到上限后，把最近一次可重试失败的原因提升为最终错误；
    # 若完全没有异常（静默失败），才用缺失产物兜底。
    if not workflow_error and state.get("attempt_error"):
        workflow_error = state["attempt_error"]
    if not workflow_error and missing:
        workflow_error = f"Create 未完成必要产物：{'、'.join(missing)}"

    return {
        "next": "FINISH",
        "workflow_error": workflow_error,
    }
