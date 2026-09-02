"""chat_session 单表 Repository。"""

from typing import Any

from databases import Database
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.models import ChatSessionModel
from app.repositories._utils import row_to_dict


class ChatSessionRepository:
    """聊天 Session 元数据的单表读写。"""

    def __init__(self, db: Database):
        self.db = db
        self.table = ChatSessionModel.__table__

    async def ensure_exists(
        self,
        *,
        session_id: str,
        user_id: int,
        title: str = "新对话",
    ) -> dict[str, Any]:
        statement = mysql_insert(self.table).values(
            session_id=session_id,
            user_id=user_id,
            title=title,
            lifecycle_status="ACTIVE",
        )
        await self.db.execute(
            statement.on_duplicate_key_update(
                session_id=statement.inserted.session_id,
            )
        )
        session = await self.get(session_id=session_id, user_id=user_id)
        if session is None:
            raise ValueError(f"Session {session_id} 已属于其他用户或无法读取")
        return session

    async def get(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(self.table).where(
                self.table.c.session_id == session_id,
                self.table.c.user_id == user_id,
            )
        )
        return row_to_dict(row)

    async def list_by_user(
        self,
        *,
        user_id: int,
        lifecycle_status: str | None = "ACTIVE",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = select(self.table).where(self.table.c.user_id == user_id)
        if lifecycle_status is not None:
            query = query.where(self.table.c.lifecycle_status == lifecycle_status)
        rows = await self.db.fetch_all(
            query.order_by(self.table.c.updated_at.desc()).limit(limit).offset(offset)
        )
        return [row_to_dict(row) or {} for row in rows]

    async def set_active_ppt(
        self,
        *,
        session_id: str,
        user_id: int,
        ppt_id: str | None,
    ) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.session_id == session_id,
                self.table.c.user_id == user_id,
                self.table.c.lifecycle_status == "ACTIVE",
            )
            .values(active_ppt_id=ppt_id)
        )
        if affected:
            return True

        # MySQL 默认按“实际发生变化的行数”返回 UPDATE rowcount。目标 PPT
        # 已经是当前 Active PPT 时 rowcount 会是 0，但这仍然是一次成功的
        # 幂等设置。回读权威记录，区分“目标状态已经满足”和“Session 不存在、
        # 不属于该用户或已经归档”。
        session = await self.get(session_id=session_id, user_id=user_id)
        return bool(
            session
            and session.get("lifecycle_status") == "ACTIVE"
            and session.get("active_ppt_id") == ppt_id
        )

    async def archive(self, *, session_id: str, user_id: int) -> bool:
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.session_id == session_id,
                self.table.c.user_id == user_id,
                self.table.c.lifecycle_status != "ARCHIVED",
            )
            .values(
                lifecycle_status="ARCHIVED",
                archived_at=func.current_timestamp(),
            )
        )
        return bool(affected)
