"""Create/Edit 共用的 PPT 记录持久化节点。"""

from pathlib import Path

from app.agents.workflow.state import WorkflowState
from app.services import PptContextService

PERSIST_PPT_RECORD_NODE = "persist_ppt_record_node"


def build_persist_node(
    ppt_context_service: PptContextService,
):
    """将 Create/Edit 的业务产物和执行进度持久化到 MySQL。

    该节点可复用：Create 子图在 Content/Assets 里程碑后各挂一个实例做
    增量持久化，最终再挂一个实例确定成功或失败；Edit 子图也复用同一节点。
    ``ppt_record`` 和 ``workflow_run`` 是这里唯一的持久化来源，不再同步
    Redis ``PptRecordStore``。
    """

    async def persist_ppt_record(state: WorkflowState) -> dict:
        ppt_id = state.get("ppt_id")
        if not ppt_id:
            return {"ppt_context_error": "缺少 PPT ID，无法保存本轮结果。"}

        user_id = state["user_id"]
        run_id = state["run_id"]
        completed_stages = list(state.get("completed_stages", []))
        required_stages = list(state.get("required_stages", []))
        state_filename = state.get("filename")
        filename = Path(state_filename).name if state_filename else None
        is_final = state.get("next") == "FINISH"
        current_stage = (
            "FINALIZE"
            if is_final
            else (completed_stages[-1].upper() if completed_stages else "INITIALIZE")
        )
        await ppt_context_service.persist_progress(
            user_id=user_id,
            run_id=run_id,
            ppt_id=ppt_id,
            current_stage=current_stage,
            completed_stages=completed_stages,
            required_stages=required_stages,
            outline=state.get("outline"),
            research_report=state.get("research_report"),
            slides_manifest=state.get("slides_manifest"),
            style=state.get("style") or "business",
            filename=filename,
        )

        if state.get("workflow_error"):
            await ppt_context_service.fail_run(
                user_id=user_id,
                run_id=run_id,
                error_code="WORKFLOW_FAILED",
                error_message=state["workflow_error"],
            )
        elif is_final:
            await ppt_context_service.complete_run(
                user_id=user_id,
                run_id=run_id,
                ppt_id=ppt_id,
                filename=filename,
                file_key=(f"ppt_output/{filename}" if filename else None),
            )
        return {}

    return persist_ppt_record


def route_after_persist(state: WorkflowState) -> str:
    return "finish" if state.get("next") == "FINISH" else "continue"
