"""Redis 客户端配置"""

import redis.asyncio as redis

from app.core.config import settings

# Redis 客户端（内部使用连接池，并在首次执行命令时建立连接）
redis_client = redis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
)
