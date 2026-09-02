"""ppt_record 单表 Repository。"""

from typing import Any

from databases import Database
from sqlalchemy import case, func, or_, select

from app.models import PptRecordModel
from app.repositories._utils import row_to_dict

_UNSET = object()


class PptRecordRepository:
    """PPT 当前业务记录的单表读写。"""

    def __init__(self, db: Database):
        self.db = db
        self.table = PptRecordModel.__table__

    async def create(
        self,
        *,
        ppt_id: str,
        owner_user_id: int,
        title: str | None = None,
        style: str = "business",
        source_type: str = "GENERATED",
    ) -> dict[str, Any]:
        await self.db.execute(
            self.table.insert().values(
                ppt_id=ppt_id,
                owner_user_id=owner_user_id,
                title=title,
                style=style,
                source_type=source_type,
                lifecycle_status="PLANNING",
                current_version=0,
            )
        )
        record = await self.get(ppt_id=ppt_id, user_id=owner_user_id)
        if record is None:
            raise RuntimeError(f"PPT 创建后无法读取: {ppt_id}")
        return record

    async def get(self, *, ppt_id: str, user_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(self.table).where(
                self.table.c.ppt_id == ppt_id,
                self.table.c.owner_user_id == user_id,
            )
        )
        return row_to_dict(row)

    async def list_by_user(
        self,
        *,
        user_id: int,
        lifecycle_status: str | None = None,
        source_type: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = select(self.table).where(self.table.c.owner_user_id == user_id)
        if lifecycle_status is not None:
            query = query.where(self.table.c.lifecycle_status == lifecycle_status)
        if source_type is not None:
            query = query.where(self.table.c.source_type == source_type)
        query = query.order_by(self.table.c.updated_at.desc())
        if limit is not None:
            query = query.limit(limit).offset(offset)
        rows = await self.db.fetch_all(query)
        return [row_to_dict(row) or {} for row in rows]

    async def update_artifacts(
        self,
        *,
        ppt_id: str,
        user_id: int,
        outline: Any = _UNSET,
        research_report: Any = _UNSET,
        slides_manifest: Any = _UNSET,
        style: Any = _UNSET,
        title: Any = _UNSET,
        filename: Any = _UNSET,
    ) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        candidates = {
            "outline_json": outline,
            "research_report_json": research_report,
            "slides_manifest_json": slides_manifest,
            "style": style,
            "title": title,
            "current_filename": filename,
        }
        for field, value in candidates.items():
            if value is not _UNSET:
                values[field] = value
        if values:
            await self.db.execute(
                self.table.update()
                .where(
                    self.table.c.ppt_id == ppt_id,
                    self.table.c.owner_user_id == user_id,
                )
                .values(**values)
            )
        return await self.get(ppt_id=ppt_id, user_id=user_id)

    async def promote_file(
        self,
        *,
        ppt_id: str,
        user_id: int,
        filename: str,
        file_key: str,
    ) -> dict[str, Any] | None:
        """将文件设为当前版本；重复提交同一 file_key 时不重复增加版本号。"""
        is_new_file = or_(
            self.table.c.current_file_key.is_(None),
            self.table.c.current_file_key != file_key,
        )
        await self.db.execute(
            self.table.update()
            .where(
                self.table.c.ppt_id == ppt_id,
                self.table.c.owner_user_id == user_id,
            )
            .values(
                current_filename=filename,
                current_file_key=file_key,
                current_version=case(
                    (is_new_file, self.table.c.current_version + 1),
                    else_=self.table.c.current_version,
                ),
                lifecycle_status="READY",
            )
        )
        return await self.get(ppt_id=ppt_id, user_id=user_id)

    async def archive(self, *, ppt_id: str, user_id: int) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.ppt_id == ppt_id,
                self.table.c.owner_user_id == user_id,
                self.table.c.lifecycle_status != "ARCHIVED",
            )
            .values(lifecycle_status="ARCHIVED", archived_at=self._now())
        )
        return bool(affected)

    @staticmethod
    def _now():
        return func.current_timestamp()
