"""Edit 的目标 PPT 解析节点。"""

from app.agents.workflow.state import WorkflowStage, WorkflowState
from app.context import PptRecord, PptRecordStore, SessionStateStore
from app.services import PptContextService, PptOwnershipError

RESOLVE_TARGET_PPT_NODE = "resolve_target_ppt_node"


def _available_stages(record: PptRecord) -> list[WorkflowStage]:
    """把已有业务产物转换成工作流可识别的完成阶段。"""
    completed: list[WorkflowStage] = []
    if record.research_report:
        completed.append("research")
    if record.outline:
        completed.append("outline")
    if record.filename:
        completed.append("content")
    return completed


def build_resolve_node(
    session_store: SessionStateStore,
    ppt_record_store: PptRecordStore,
    ppt_context_service: PptContextService | None = None,
):
    """显式 ppt_id 优先，否则使用活动 PPT；MySQL 按用户校验所有权。"""

    async def resolve_target_ppt(state: WorkflowState) -> dict:
        session_state = await session_store.load(state["session_id"])
        requested_ppt_id = state.get("requested_ppt_id")
        target_ppt_id = requested_ppt_id or session_state.active_ppt_id

        if ppt_context_service is not None:
            if not target_ppt_id:
                await ppt_context_service.initialize_edit(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    run_id=state["run_id"],
                    message=state["user_message"],
                    requested_ppt_id=None,
                    checkpoint_thread_id=state["run_id"],
                )
                return {
                    "ppt_id": None,
                    "ppt_context_error": "当前会话还没有可编辑的 PPT，请选择或上传一份 PPT。",
                }

            try:
                initialized = await ppt_context_service.initialize_edit(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    run_id=state["run_id"],
                    message=state["user_message"],
                    requested_ppt_id=target_ppt_id,
                    checkpoint_thread_id=state["run_id"],
                )
            except PptOwnershipError:
                await ppt_context_service.initialize_edit(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    run_id=state["run_id"],
                    message=state["user_message"],
                    requested_ppt_id=None,
                    waiting_payload={"requested_ppt_id": target_ppt_id},
                    checkpoint_thread_id=state["run_id"],
                )
                return {
                    "ppt_id": None,
                    "ppt_context_error": "指定的 PPT 不存在或不属于当前用户，请重新选择或上传。",
                }

            mysql_record = initialized["ppt"]
            record = PptRecord(
                ppt_id=target_ppt_id,
                filename=mysql_record.get("current_filename"),
                slides_manifest=mysql_record.get("slides_manifest_json"),
                outline=mysql_record.get("outline_json"),
                research_report=mysql_record.get("research_report_json"),
                style=mysql_record.get("style") or "business",
                status=(
                    "completed"
                    if mysql_record.get("lifecycle_status") == "READY"
                    else "in_progress"
                ),
            )
            await ppt_record_store.save(record)
            await session_store.add_ppt(state["session_id"], target_ppt_id)
            return {
                "ppt_id": target_ppt_id,
                "active_ppt_id": target_ppt_id,
                "ppt_context_error": None,
                "outline": record.outline,
                "research_report": record.research_report,
                "filename": record.filename,
                "slides_manifest": record.slides_manifest,
                "style": record.style,
                "asset_operations": [],
                "asset_apply_status": "pending",
                "applied_operation_ids": [],
                "asset_tasks": [],
                "required_stages": [],
                "completed_stages": _available_stages(record),
                "requirements_initialized": False,
                "next": None,
            }

        if not target_ppt_id:
            return {
                "ppt_id": None,
                "ppt_context_error": "当前会话还没有可编辑的 PPT。",
            }
        if target_ppt_id not in session_state.ppt_ids:
            return {
                "ppt_id": None,
                "ppt_context_error": "指定的 PPT 不属于当前会话，无法编辑。",
            }

        record = await ppt_record_store.load(target_ppt_id)
        if record is None:
            return {
                "ppt_id": None,
                "ppt_context_error": "目标 PPT 记录不存在或无法读取。",
            }

        await session_store.set_active_ppt(state["session_id"], target_ppt_id)
        return {
            "ppt_id": target_ppt_id,
            "active_ppt_id": target_ppt_id,
            "ppt_context_error": None,
            "outline": record.outline,
            "research_report": record.research_report,
            "filename": record.filename,
            "slides_manifest": record.slides_manifest,
            "style": record.style,
            "asset_operations": [],
            "asset_apply_status": "pending",
            "applied_operation_ids": [],
            "asset_tasks": [],
            "required_stages": [],
            "completed_stages": _available_stages(record),
            "requirements_initialized": False,
            "next": None,
        }

    return resolve_target_ppt


def route_after_resolution(state: WorkflowState) -> str:
    return "error" if state.get("ppt_context_error") else "resolved"
