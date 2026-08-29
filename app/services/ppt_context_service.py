"""PPT、Session 与 Workflow Run 的跨表事务服务。"""

from typing import Any

from databases import Database

from app.repositories import (
    ChatSessionRepository,
    PptRecordRepository,
    SessionPptRepository,
    WorkflowRunRepository,
)


class PptOwnershipError(ValueError):
    """PPT 不存在或不属于当前用户。"""


class WorkflowRunConflictError(RuntimeError):
    """Workflow Run 状态或 revision 已被其他请求推进。"""


class PptContextService:
    """协调四张持久化表，并保证一次业务变更在同一事务中完成。"""

    def __init__(
        self,
        db: Database,
        *,
        ppt_repository: PptRecordRepository | None = None,
        session_repository: ChatSessionRepository | None = None,
        session_ppt_repository: SessionPptRepository | None = None,
        run_repository: WorkflowRunRepository | None = None,
    ):
        self.db = db
        self.ppt_repository = ppt_repository or PptRecordRepository(db)
        self.session_repository = session_repository or ChatSessionRepository(db)
        self.session_ppt_repository = session_ppt_repository or SessionPptRepository(db)
        self.run_repository = run_repository or WorkflowRunRepository(db)

    async def ensure_session(
        self,
        *,
        session_id: str,
        user_id: int,
        title: str = "新对话",
    ) -> dict[str, Any]:
        return await self.session_repository.ensure_exists(
            session_id=session_id,
            user_id=user_id,
            title=title,
        )

    async def initialize_create(
        self,
        *,
        user_id: int,
        session_id: str,
        run_id: str,
        ppt_id: str,
        message: str,
        style: str,
        title: str | None = None,
        checkpoint_thread_id: str | None = None,
        graph_version: str = "v1",
        required_stages: list[str] | None = None,
    ) -> dict[str, Any]:
        """原子创建 Session、PPT、Session 引用和 Create Run。"""
        stages = required_stages or ["research", "outline", "content"]
        async with self.db.transaction():
            await self.session_repository.ensure_exists(
                session_id=session_id,
                user_id=user_id,
                title=title or self._default_session_title(message),
            )
            ppt = await self.ppt_repository.create(
                ppt_id=ppt_id,
                owner_user_id=user_id,
                title=title,
                style=style,
                source_type="GENERATED",
            )
            await self.session_ppt_repository.link(
                session_id=session_id,
                ppt_id=ppt_id,
                association_source="CREATED",
            )
            if not await self.session_repository.set_active_ppt(
                session_id=session_id,
                user_id=user_id,
                ppt_id=ppt_id,
            ):
                raise RuntimeError(f"无法设置 Session 活动 PPT: {session_id}")
            run = await self.run_repository.create(
                run_id=run_id,
                user_id=user_id,
                session_id=session_id,
                ppt_id=ppt_id,
                intent="CREATE",
                status="RUNNING",
                required_stages=stages,
                checkpoint_thread_id=checkpoint_thread_id or run_id,
                graph_version=graph_version,
                input_payload={
                    "message": message,
                    "style": style,
                    "requested_ppt_id": None,
                },
            )
        return {"ppt": ppt, "run": run}

    async def initialize_edit(
        self,
        *,
        user_id: int,
        session_id: str,
        run_id: str,
        message: str,
        requested_ppt_id: str | None = None,
        waiting_payload: dict[str, Any] | None = None,
        checkpoint_thread_id: str | None = None,
        graph_version: str = "v1",
    ) -> dict[str, Any]:
        """创建 Edit Run；目标未知时在同一事务中进入 HITL 等待状态。"""
        async with self.db.transaction():
            await self.session_repository.ensure_exists(
                session_id=session_id,
                user_id=user_id,
                title=self._default_session_title(message),
            )
            ppt = None
            if requested_ppt_id is not None:
                ppt = await self.ppt_repository.get(
                    ppt_id=requested_ppt_id,
                    user_id=user_id,
                )
                if ppt is None:
                    raise PptOwnershipError(
                        f"PPT 不存在或不属于当前用户: {requested_ppt_id}"
                    )
                await self.session_ppt_repository.link(
                    session_id=session_id,
                    ppt_id=requested_ppt_id,
                    association_source="SELECTED",
                )
                if not await self.session_repository.set_active_ppt(
                    session_id=session_id,
                    user_id=user_id,
                    ppt_id=requested_ppt_id,
                ):
                    raise RuntimeError(f"无法设置 Session 活动 PPT: {session_id}")

            run = await self.run_repository.create(
                run_id=run_id,
                user_id=user_id,
                session_id=session_id,
                ppt_id=requested_ppt_id,
                intent="EDIT",
                status="RUNNING",
                required_stages=[],
                checkpoint_thread_id=checkpoint_thread_id or run_id,
                graph_version=graph_version,
                input_payload={
                    "message": message,
                    "requested_ppt_id": requested_ppt_id,
                },
            )
            if requested_ppt_id is None:
                if not await self.run_repository.mark_waiting(
                    run_id=run_id,
                    user_id=user_id,
                    waiting_type="PPT_TARGET_REQUIRED",
                    waiting_payload=waiting_payload or {"requested_ppt_id": None},
                    current_stage="RESOLVE_TARGET",
                ):
                    raise WorkflowRunConflictError(
                        f"Workflow Run 无法进入等待状态: {run_id}"
                    )
                refreshed_run = await self.run_repository.get(
                    run_id=run_id,
                    user_id=user_id,
                )
                if refreshed_run is None:
                    raise RuntimeError(f"Workflow Run 等待状态无法读取: {run_id}")
                run = refreshed_run
        return {"ppt": ppt, "run": run}

    async def bind_edit_target(
        self,
        *,
        user_id: int,
        run_id: str,
        ppt_id: str,
        expected_revision: int,
        association_source: str = "SELECTED",
    ) -> dict[str, Any]:
        """用户在 HITL 中选择/上传 PPT 后，原子绑定并恢复 Run。"""
        async with self.db.transaction():
            run = await self.run_repository.get(run_id=run_id, user_id=user_id)
            if run is None:
                raise WorkflowRunConflictError(f"Workflow Run 不存在: {run_id}")
            ppt = await self.ppt_repository.get(ppt_id=ppt_id, user_id=user_id)
            if ppt is None:
                raise PptOwnershipError(f"PPT 不存在或不属于当前用户: {ppt_id}")

            session_id = str(run["session_id"])
            await self.session_ppt_repository.link(
                session_id=session_id,
                ppt_id=ppt_id,
                association_source=association_source,
            )
            if not await self.session_repository.set_active_ppt(
                session_id=session_id,
                user_id=user_id,
                ppt_id=ppt_id,
            ):
                raise RuntimeError(f"无法设置 Session 活动 PPT: {session_id}")
            if not await self.run_repository.bind_ppt_and_resume(
                run_id=run_id,
                user_id=user_id,
                ppt_id=ppt_id,
                expected_revision=expected_revision,
            ):
                raise WorkflowRunConflictError(
                    f"Workflow Run 已被其他请求恢复或 revision 不匹配: {run_id}"
                )
        return {"ppt": ppt, "run_id": run_id, "session_id": session_id}

    async def load_create_resume_context(
        self,
        *,
        user_id: int,
        run_id: str,
    ) -> dict[str, dict[str, Any]]:
        """通过可恢复的 Create Run 加载对应 PPT 业务快照。"""
        run = await self.run_repository.get(run_id=run_id, user_id=user_id)
        if run is None:
            raise WorkflowRunConflictError(
                f"Workflow Run 不存在或不属于当前用户: {run_id}"
            )
        if str(run.get("intent", "")).upper() != "CREATE":
            raise WorkflowRunConflictError(
                f"Workflow Run 不是 Create 类型，无法断点重跑: {run_id}"
            )
        if run.get("run_status") != "RUNNING":
            raise WorkflowRunConflictError(
                f"Workflow Run 当前状态不可恢复: {run_id} ({run.get('run_status')})"
            )

        ppt_id = run.get("ppt_id")
        if not isinstance(ppt_id, str) or not ppt_id:
            raise WorkflowRunConflictError(
                f"Workflow Run 未关联 PPT，无法断点重跑: {run_id}"
            )
        ppt = await self.ppt_repository.get(ppt_id=ppt_id, user_id=user_id)
        if ppt is None:
            raise PptOwnershipError(f"PPT 不存在或不属于当前用户: {ppt_id}")
        return {"run": run, "ppt": ppt}

    async def list_create_tasks(
        self,
        *,
        user_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """返回用户每份 PPT 最新一次 Create Run 的简洁任务视图。"""
        runs = await self.run_repository.list_by_user(
            user_id=user_id,
            intent="CREATE",
            limit=limit,
        )
        tasks: list[dict[str, Any]] = []
        seen_ppt_ids: set[str] = set()
        for run in runs:
            ppt_id = run.get("ppt_id")
            if not isinstance(ppt_id, str) or not ppt_id or ppt_id in seen_ppt_ids:
                continue
            seen_ppt_ids.add(ppt_id)
            ppt = await self.ppt_repository.get(ppt_id=ppt_id, user_id=user_id)
            if ppt is None:
                continue
            input_payload = run.get("input_payload_json") or {}
            tasks.append(
                {
                    "run_id": run["run_id"],
                    "ppt_id": ppt_id,
                    "title": (
                        ppt.get("title")
                        or input_payload.get("message")
                        or "未命名 PPT"
                    ),
                    "style": ppt.get("style") or "business",
                    "run_status": run.get("run_status"),
                    "current_stage": run.get("current_stage"),
                    "completed_stages": run.get("completed_stages_json") or [],
                    "required_stages": run.get("required_stages_json") or [],
                    "filename": ppt.get("current_filename"),
                    "error_message": run.get("error_message"),
                    "updated_at": run.get("updated_at") or ppt.get("updated_at"),
                }
            )
        return tasks

    async def persist_progress(
        self,
        *,
        user_id: int,
        run_id: str,
        ppt_id: str,
        current_stage: str,
        completed_stages: list[str],
        required_stages: list[str],
        outline: Any = None,
        research_report: Any = None,
        slides_manifest: Any = None,
        style: str | None = None,
        filename: str | None = None,
    ) -> None:
        """在同一事务中更新 PPT 阶段产物和 Run 进度。"""
        async with self.db.transaction():
            ppt = await self.ppt_repository.get(ppt_id=ppt_id, user_id=user_id)
            if ppt is None:
                raise PptOwnershipError(f"PPT 不存在或不属于当前用户: {ppt_id}")
            updated = await self.ppt_repository.update_artifacts(
                ppt_id=ppt_id,
                user_id=user_id,
                outline=outline if outline is not None else ppt.get("outline_json"),
                research_report=(
                    research_report
                    if research_report is not None
                    else ppt.get("research_report_json")
                ),
                slides_manifest=(
                    slides_manifest
                    if slides_manifest is not None
                    else ppt.get("slides_manifest_json")
                ),
                style=style if style is not None else ppt.get("style"),
                filename=(
                    filename if filename is not None else ppt.get("current_filename")
                ),
            )
            if updated is None:
                raise PptOwnershipError(f"PPT 更新失败: {ppt_id}")
            if not await self.run_repository.update_progress(
                run_id=run_id,
                user_id=user_id,
                current_stage=current_stage,
                completed_stages=completed_stages,
                required_stages=required_stages,
            ):
                raise WorkflowRunConflictError(f"Workflow Run 不在可更新状态: {run_id}")

    async def mark_waiting(
        self,
        *,
        user_id: int,
        run_id: str,
        waiting_type: str,
        waiting_payload: dict[str, Any],
        current_stage: str,
    ) -> None:
        if not await self.run_repository.mark_waiting(
            run_id=run_id,
            user_id=user_id,
            waiting_type=waiting_type,
            waiting_payload=waiting_payload,
            current_stage=current_stage,
        ):
            raise WorkflowRunConflictError(f"Workflow Run 无法进入等待状态: {run_id}")

    async def complete_run(
        self,
        *,
        user_id: int,
        run_id: str,
        ppt_id: str,
        filename: str | None = None,
        file_key: str | None = None,
    ) -> None:
        """原子发布最终文件（如有）并把 Run 标记为成功。"""
        async with self.db.transaction():
            if filename is not None or file_key is not None:
                if not filename or not file_key:
                    raise ValueError("filename 与 file_key 必须同时提供")
                ppt = await self.ppt_repository.promote_file(
                    ppt_id=ppt_id,
                    user_id=user_id,
                    filename=filename,
                    file_key=file_key,
                )
                if ppt is None:
                    raise PptOwnershipError(f"PPT 文件发布失败: {ppt_id}")
            if not await self.run_repository.mark_succeeded(
                run_id=run_id,
                user_id=user_id,
            ):
                raise WorkflowRunConflictError(f"Workflow Run 无法完成: {run_id}")

    async def fail_run(
        self,
        *,
        user_id: int,
        run_id: str,
        error_message: str,
        error_code: str | None = None,
    ) -> None:
        """只标记本次 Run 失败，不破坏已有 PPT 的 READY 状态。"""
        if not await self.run_repository.mark_failed(
            run_id=run_id,
            user_id=user_id,
            error_code=error_code,
            error_message=error_message,
        ):
            raise WorkflowRunConflictError(f"Workflow Run 无法标记失败: {run_id}")

    @staticmethod
    def _default_session_title(message: str) -> str:
        title = " ".join(message.split())
        return title[:255] or "新对话"
