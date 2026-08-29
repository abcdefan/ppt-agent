"""chat_message 单表 Repository。"""

from typing import Any

from databases import Database
from sqlalchemy import select

from app.models import ChatMessageModel
from app.repositories._utils import row_to_dict


class ChatMessageRepository:
    """用户和 Assistant 最终可见消息的单表读写。"""

    def __init__(self, db: Database):
        self.db = db
        self.table = ChatMessageModel.__table__

    async def create(
        self,
        *,
        session_id: str,
        message_role: str,
        content: str,
        run_id: str | None = None,
        message_type: str = "TEXT",
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
        message_status: str = "COMPLETED",
    ) -> dict[str, Any]:
        message_id = await self.db.execute(
            self.table.insert().values(
                session_id=session_id,
                run_id=run_id,
                message_role=message_role,
                message_type=message_type,
                content=content,
                metadata_json=metadata,
                token_count=token_count,
                message_status=message_status,
            )
        )
        message = await self.get(message_id=int(message_id))
        if message is None:
            raise RuntimeError(f"消息创建后无法读取: {message_id}")
        return message

    async def get(self, *, message_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(self.table).where(self.table.c.id == message_id)
        )
        return row_to_dict(row)

    async def attach_run(self, *, message_id: int, run_id: str) -> bool:
        """Intent 确认且 Run 已创建后，为用户消息补充运行关联。"""
        affected = await self.db.execute(
            self.table.update()
            .where(
                self.table.c.id == message_id,
                self.table.c.run_id.is_(None),
            )
            .values(run_id=run_id)
        )
        return bool(affected)

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
            .order_by(self.table.c.id.asc())
            .limit(limit)
            .offset(offset)
        )
        return [row_to_dict(row) or {} for row in rows]

    async def list_recent_by_session(
        self,
        *,
        session_id: str,
        limit: int = 20,
        before_id: int | None = None,
    ) -> list[dict[str, Any]]:
        query = select(self.table).where(self.table.c.session_id == session_id)
        if before_id is not None:
            query = query.where(self.table.c.id < before_id)
        rows = await self.db.fetch_all(
            query.order_by(self.table.c.id.desc()).limit(limit)
        )
        return list(reversed([row_to_dict(row) or {} for row in rows]))
