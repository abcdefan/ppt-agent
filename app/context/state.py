"""结构化 Session State 模型及其 Redis 持久化。"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core import redis as redis_module

logger = logging.getLogger(__name__)

_STATE_PREFIX = "agent:session_state:"
_PERSISTENT_FIELDS = {
    "active_ppt_filename",
    "outline",
    "style",
}


class SessionState(BaseModel):
    """当前会话正在操作的 PPT 业务上下文。"""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(default=1, ge=1)
    active_ppt_filename: str | None = None
    outline: str | None = None
    style: str = "business"


class SessionStateStore:
    """按 session_id 完整加载、白名单保存业务 State。"""

    def __init__(self, redis_client=None):
        self._redis_client = redis_client

    @property
    def redis(self):
        return self._redis_client or redis_module.redis_client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{_STATE_PREFIX}{session_id}"

    async def load(self, session_id: str) -> SessionState:
        client = self.redis
        if not client:
            return SessionState()

        try:
            raw = await client.get(self._key(session_id))
            if not raw:
                return SessionState()
            return SessionState.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("[SessionState] 会话 %s 数据损坏，使用默认值: %s", session_id, exc)
            return SessionState()
        except Exception as exc:
            logger.warning("[SessionState] 会话 %s 加载失败，使用默认值: %s", session_id, exc)
            return SessionState()

    async def save(self, session_id: str, state: SessionState) -> None:
        client = self.redis
        if not client:
            return
        await client.set(self._key(session_id), state.model_dump_json())

    async def patch(self, session_id: str, **updates: Any) -> SessionState:
        """只允许更新明确列入持久化白名单的字段。"""
        safe_updates = {
            key: value for key, value in updates.items() if key in _PERSISTENT_FIELDS
        }
        state = await self.load(session_id)
        if not safe_updates:
            return state

        updated = state.model_copy(update=safe_updates)
        # model_copy(update=...) 不重新校验；重新构造以保持持久化数据合法。
        updated = SessionState.model_validate(updated.model_dump())
        await self.save(session_id, updated)
        return updated

    async def clear(self, session_id: str) -> None:
        client = self.redis
        if client:
            await client.delete(self._key(session_id))
