"""Create 的 PPT 上下文初始化节点。"""

from uuid import uuid4

from app.agents.workflow.state import WorkflowState
from app.context import PptRecord, PptRecordStore, SessionStateStore
from app.services import PptContextService

INITIALIZE_PPT_NODE = "initialize_ppt_node"


def build_initialize_node(
    session_store: SessionStateStore,
    ppt_record_store: PptRecordStore,
    ppt_context_service: PptContextService | None = None,
):
    """为 Create 创建 MySQL 业务记录，并同步 Redis 运行时缓存。"""

    async def initialize_ppt(state: WorkflowState) -> dict:
        ppt_id = uuid4().hex
        style = state.get("style") or "business"
        if ppt_context_service is not None:
            await ppt_context_service.initialize_create(
                user_id=state["user_id"],
                session_id=state["session_id"],
                run_id=state["run_id"],
                ppt_id=ppt_id,
                message=state["user_message"],
                style=style,
                title=state["user_message"][:255],
                checkpoint_thread_id=state["run_id"],
                required_stages=["research", "outline", "content"],
            )
        await ppt_record_store.save(
            PptRecord(
                ppt_id=ppt_id,
                style=style,
                status="planning",
            )
        )
        await session_store.add_ppt(state["session_id"], ppt_id)
        return {
            "ppt_id": ppt_id,
            "active_ppt_id": ppt_id,
            "ppt_context_error": None,
            "workflow_error": None,
            "research_report": None,
            "outline": None,
            "filename": None,
            "slides_manifest": None,
            "style": style,
            "asset_operations": [],
            "asset_apply_status": "pending",
            "applied_operation_ids": [],
            "asset_tasks": [],
            # Create 的基础阶段由代码确定；可选增强阶段等 Research 和
            # Content 完成并写回页面清单后，再由 Enhancement Planner 追加
            # 可选增强阶段并冻结。
            "required_stages": ["research", "outline", "content"],
            "requirements_initialized": False,
            "create_finalized": False,
            "attempt_error": None,
            "attempt_counts": {},
            "edit_next": None,
        }

    return initialize_ppt
