"""用于锁实验的 Redis PPT 锁 Tools。

这些 Tool 故意不注册给任何 Agent，也不接入默认 Workflow。生产代码若要
使用分布式锁，应优先在确定性 edit_node 内用上下文管理器封装完整生命周期。
"""

import asyncio
import hashlib
import json
import time
import uuid

from langchain_core.tools import tool

from app.core.redis import redis_client

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def _lock_key(ppt_id: str) -> str:
    """避免把用户输入直接拼进 Redis Key。"""
    normalized = ppt_id.strip()
    if not normalized:
        raise ValueError("ppt_id 不能为空")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"ppt:write-lock:{digest}"


@tool
async def acquire_ppt_lock(
    ppt_id: str,
    ttl_seconds: int = 60,
    blocking_timeout_seconds: float = 10.0,
) -> str:
    """获取以 ppt_id 为粒度的 Redis 写锁，返回释放锁所需的 owner_token。

    注意：此 Tool 仅用于锁等待和 Trace 实验，当前没有注册给任何 Agent。

    Args:
        ppt_id: PPT 的稳定业务 ID，不是展示文件名
        ttl_seconds: 锁自动过期时间
        blocking_timeout_seconds: 获取锁的最大等待时间
    """
    if ttl_seconds <= 0:
        return json.dumps(
            {"success": False, "message": "ttl_seconds 必须大于 0"},
            ensure_ascii=False,
        )
    if blocking_timeout_seconds < 0:
        return json.dumps(
            {
                "success": False,
                "message": "blocking_timeout_seconds 不能小于 0",
            },
            ensure_ascii=False,
        )

    key = _lock_key(ppt_id)
    owner_token = uuid.uuid4().hex
    started_at = time.monotonic()

    while True:
        acquired = await redis_client.set(
            key,
            owner_token,
            nx=True,
            ex=ttl_seconds,
        )
        if acquired:
            return json.dumps(
                {
                    "success": True,
                    "lock_key": key,
                    "owner_token": owner_token,
                    "ttl_seconds": ttl_seconds,
                    "waited_ms": round((time.monotonic() - started_at) * 1000, 2),
                },
                ensure_ascii=False,
            )

        elapsed = time.monotonic() - started_at
        if elapsed >= blocking_timeout_seconds:
            return json.dumps(
                {
                    "success": False,
                    "code": "PPT_LOCK_TIMEOUT",
                    "lock_key": key,
                    "waited_ms": round(elapsed * 1000, 2),
                    "message": "等待 PPT Redis 写锁超时",
                },
                ensure_ascii=False,
            )
        await asyncio.sleep(min(0.1, blocking_timeout_seconds - elapsed))


@tool
async def release_ppt_lock(ppt_id: str, owner_token: str) -> str:
    """仅当 owner_token 匹配时释放 Redis PPT 写锁。

    注意：此 Tool 仅用于锁等待和 Trace 实验，当前没有注册给任何 Agent。

    Args:
        ppt_id: 获取锁时使用的同一个 PPT 业务 ID
        owner_token: acquire_ppt_lock 返回的所有者令牌
    """
    if not owner_token:
        return json.dumps(
            {"success": False, "message": "owner_token 不能为空"},
            ensure_ascii=False,
        )

    key = _lock_key(ppt_id)
    released = await redis_client.eval(
        _RELEASE_SCRIPT,
        1,
        key,
        owner_token,
    )
    return json.dumps(
        {
            "success": bool(released),
            "lock_key": key,
            "message": (
                "PPT Redis 写锁已释放"
                if released
                else "锁不存在、已过期或 owner_token 不匹配"
            ),
        },
        ensure_ascii=False,
    )
