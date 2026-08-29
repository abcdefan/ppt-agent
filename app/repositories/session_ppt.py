"""session_ppt 单表 Repository。"""

from typing import Any

from databases import Database
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.models import SessionPptModel
from app.repositories._utils import row_to_dict


class SessionPptRepository:
    """Session 与 PPT 引用关系的单表读写。"""

    def __init__(self, db: Database):
        self.db = db
        self.table = SessionPptModel.__table__

    async def link(
        self,
        *,
        session_id: str,
        ppt_id: str,
        association_source: str,
    ) -> None:
        statement = mysql_insert(self.table).values(
            session_id=session_id,
            ppt_id=ppt_id,
            association_source=association_source,
        )
        await self.db.execute(
            statement.on_duplicate_key_update(
                last_used_at=func.current_timestamp(),
            )
        )

    async def touch(self, *, session_id: str, ppt_id: str) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.session_id == session_id,
                self.table.c.ppt_id == ppt_id,
            )
            .values(last_used_at=func.current_timestamp())
        )
        return bool(affected)

    async def exists(self, *, session_id: str, ppt_id: str) -> bool:
        row = await self.db.fetch_one(
            select(self.table.c.session_id).where(
                self.table.c.session_id == session_id,
                self.table.c.ppt_id == ppt_id,
            )
        )
        return row is not None

    async def list_by_session(
        self,
        *,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            select(self.table)
            .where(self.table.c.session_id == session_id)
            .order_by(self.table.c.last_used_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [row_to_dict(row) or {} for row in rows]
