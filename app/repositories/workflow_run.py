"""workflow_run 单表 Repository。"""

from typing import Any

from databases import Database
from sqlalchemy import func, select

from app.models import WorkflowRunModel
from app.repositories._utils import row_to_dict


class WorkflowRunRepository:
    """Workflow 单次执行记录的单表读写。"""

    def __init__(self, db: Database):
        self.db = db
        self.table = WorkflowRunModel.__table__

    async def create(
        self,
        *,
        run_id: str,
        user_id: int,
        session_id: str,
        intent: str,
        input_payload: dict[str, Any],
        required_stages: list[str],
        ppt_id: str | None = None,
        checkpoint_thread_id: str | None = None,
        graph_version: str = "v1",
        status: str = "RUNNING",
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "run_id": run_id,
            "user_id": user_id,
            "session_id": session_id,
            "ppt_id": ppt_id,
            "intent": intent,
            "run_status": status,
            "completed_stages_json": [],
            "required_stages_json": required_stages,
            "checkpoint_thread_id": checkpoint_thread_id or run_id,
            "graph_version": graph_version,
            "input_payload_json": input_payload,
        }
        if status == "RUNNING":
            values["started_at"] = func.current_timestamp()
        await self.db.execute(self.table.insert().values(**values))
        run = await self.get(run_id=run_id, user_id=user_id)
        if run is None:
            raise RuntimeError(f"Workflow Run 创建后无法读取: {run_id}")
        return run

    async def get(self, *, run_id: str, user_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(self.table).where(
                self.table.c.run_id == run_id,
                self.table.c.user_id == user_id,
            )
        )
        return row_to_dict(row)

    async def list_by_user(
        self,
        *,
        user_id: int,
        status: str | None = None,
        intent: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = select(self.table).where(self.table.c.user_id == user_id)
        if status is not None:
            query = query.where(self.table.c.run_status == status)
        if intent is not None:
            query = query.where(self.table.c.intent == intent)
        rows = await self.db.fetch_all(
            query.order_by(self.table.c.updated_at.desc()).limit(limit).offset(offset)
        )
        return [row_to_dict(row) or {} for row in rows]

    async def update_progress(
        self,
        *,
        run_id: str,
        user_id: int,
        current_stage: str,
        completed_stages: list[str],
        required_stages: list[str],
    ) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.run_id == run_id,
                self.table.c.user_id == user_id,
                self.table.c.run_status.in_(["CREATED", "RUNNING"]),
            )
            .values(
                run_status="RUNNING",
                current_stage=current_stage,
                completed_stages_json=completed_stages,
                required_stages_json=required_stages,
                revision=self.table.c.revision + 1,
            )
        )
        return bool(affected)

    async def mark_waiting(
        self,
        *,
        run_id: str,
        user_id: int,
        waiting_type: str,
        waiting_payload: dict[str, Any],
        current_stage: str,
    ) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.run_id == run_id,
                self.table.c.user_id == user_id,
                self.table.c.run_status.in_(["CREATED", "RUNNING"]),
            )
            .values(
                run_status="WAITING_INPUT",
                current_stage=current_stage,
                waiting_type=waiting_type,
                waiting_payload_json=waiting_payload,
                waiting_since=func.current_timestamp(),
                revision=self.table.c.revision + 1,
            )
        )
        return bool(affected)

    async def bind_ppt_and_resume(
        self,
        *,
        run_id: str,
        user_id: int,
        ppt_id: str,
        expected_revision: int,
    ) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.run_id == run_id,
                self.table.c.user_id == user_id,
                self.table.c.run_status == "WAITING_INPUT",
                self.table.c.waiting_type == "PPT_TARGET_REQUIRED",
                self.table.c.revision == expected_revision,
            )
            .values(
                ppt_id=ppt_id,
                run_status="RUNNING",
                waiting_type=None,
                waiting_payload_json=None,
                waiting_since=None,
                revision=self.table.c.revision + 1,
            )
        )
        return bool(affected)

    async def mark_succeeded(self, *, run_id: str, user_id: int) -> bool:
        return await self._mark_terminal(
            run_id=run_id,
            user_id=user_id,
            status="SUCCEEDED",
        )

    async def mark_failed(
        self,
        *,
        run_id: str,
        user_id: int,
        error_message: str,
        error_code: str | None = None,
    ) -> bool:
        return await self._mark_terminal(
            run_id=run_id,
            user_id=user_id,
            status="FAILED",
            error_code=error_code,
            error_message=error_message,
        )

    async def _mark_terminal(
        self,
        *,
        run_id: str,
        user_id: int,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.run_id == run_id,
                self.table.c.user_id == user_id,
                self.table.c.run_status.in_(["CREATED", "RUNNING"]),
            )
            .values(
                run_status=status,
                current_stage="FINALIZE",
                waiting_type=None,
                waiting_payload_json=None,
                waiting_since=None,
                error_code=error_code,
                error_message=error_message,
                finished_at=func.current_timestamp(),
                revision=self.table.c.revision + 1,
            )
        )
        return bool(affected)
