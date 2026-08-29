"""PPT 业务记录及其 Redis 持久化。"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core import redis as redis_module

logger = logging.getLogger(__name__)

_PPT_RECORD_PREFIX = "agent:ppt_record:"
_PERSISTENT_FIELDS = {
    "filename",
    "slides_manifest",
    "outline",
    "research_report",
    "style",
    "workflow_error",
    "status",
}

PptStatus = Literal["planning", "in_progress", "completed", "failed"]


class PptRecord(BaseModel):
    """一份 PPT 的 Redis 业务记录。"""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(default=1, ge=1)
    ppt_id: str = Field(min_length=1)
    filename: str | None = None
    slides_manifest: list[dict[str, Any]] | None = None
    outline: str | None = None
    research_report: str | None = None
    style: str = "business"
    workflow_error: str | None = None
    status: PptStatus = "planning"


class PptRecordStore:
    """按 ppt_id 加载和白名单更新 PPT 记录。"""

    def __init__(self, redis_client=None):
        self._redis_client = redis_client

    @property
    def redis(self):
        return self._redis_client or redis_module.redis_client

    @staticmethod
    def _key(ppt_id: str) -> str:
        return f"{_PPT_RECORD_PREFIX}{ppt_id}"

    async def load(self, ppt_id: str) -> PptRecord | None:
        client = self.redis
        if not client:
            return None
        try:
            raw = await client.get(self._key(ppt_id))
            if not raw:
                return None
            return PptRecord.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("[PptRecord] PPT %s 数据损坏: %s", ppt_id, exc)
            return None
        except Exception as exc:
            logger.warning("[PptRecord] PPT %s 加载失败: %s", ppt_id, exc)
            return None

    async def save(self, record: PptRecord) -> None:
        client = self.redis
        if client:
            await client.set(self._key(record.ppt_id), record.model_dump_json())

    async def patch(self, ppt_id: str, **updates: Any) -> PptRecord:
        record = await self.load(ppt_id)
        if record is None:
            raise ValueError(f"PPT 记录不存在: {ppt_id}")
        safe_updates = {
            key: value for key, value in updates.items() if key in _PERSISTENT_FIELDS
        }
        if not safe_updates:
            return record
        updated = PptRecord.model_validate(
            record.model_copy(update=safe_updates).model_dump()
        )
        await self.save(updated)
        return updated

    async def clear(self, ppt_id: str) -> None:
        client = self.redis
        if client:
            await client.delete(self._key(ppt_id))
