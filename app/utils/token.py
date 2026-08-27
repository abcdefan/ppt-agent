"""Token 管理工具。

本模块使用“随机 Token ID + Redis”保存用户的登录状态：
客户端只持有随机生成的 Token ID，具体的用户信息保存在服务端 Redis 中。
"""

import json
import uuid

from app.core import redis as redis_module
from app.core.config import settings


def generate_token_id() -> str:
    """生成一个随机且唯一的 Token ID。

    用户登录成功后，后端会调用此函数生成登录凭证。这个 ID 会返回给
    客户端，并作为 Redis 中登录信息的一部分 Key。UUID4 基于随机数生成，
    用户无法像修改连续的用户 ID 那样轻易猜出其他用户的有效 Token。

    Returns:
        字符串形式的 UUID，例如
        ``550e8400-e29b-41d4-a716-446655440000``。
    """
    return str(uuid.uuid4())


def _get_token_key(token_id: str) -> str:
    """把 Token ID 转换成 Redis 中实际使用的 Key。

    统一添加 ``token:`` 前缀，可以区分 Redis 中的 Token 数据和其他业务
    数据，也方便后续查找、监控或清理所有 Token。函数名以单下划线开头，
    表示它只供当前模块内部使用。

    Args:
        token_id: 客户端提交的 Token ID。

    Returns:
        带命名空间前缀的 Redis Key，例如 ``token:550e8400-...``。
    """
    return f"token:{token_id}"


async def get_token(token_id: str) -> dict | None:
    """根据 Token ID 从 Redis 读取用户登录信息。

    后端收到需要登录的请求时，可以先调用此函数验证 Token。Redis 中有
    对应数据，说明 Token 当前有效；不存在则可能是 Token 错误、已经退出
    登录或已经超过有效期。

    Args:
        token_id: 客户端随请求携带的 Token ID。

    Returns:
        Token 有效时返回反序列化后的用户信息字典，例如
        ``{"user_id": 1, "username": "alice"}``；Redis 客户端未创建或
        Token 不存在时返回 ``None``。Redis 服务连接异常时仍会抛出异常。
    """
    if not redis_module.redis_client:
        return None

    key = _get_token_key(token_id)
    # Redis 保存的是 JSON 字符串，需要先读取再还原成 Python 字典。
    data = await redis_module.redis_client.get(key)

    if data:
        return json.loads(data)
    return None


async def set_token(token_id: str, data: dict, expire: int | None = None):
    """把 Token 对应的用户登录信息写入 Redis，并设置有效期。

    一般在用户名和密码验证成功后调用。``setex`` 会同时保存数据和设置
    过期时间，超过有效期后 Redis 会自动删除该 Token，用户需要重新登录。

    Args:
        token_id: ``generate_token_id`` 生成的随机 Token ID。
        data: 与 Token 关联的可信用户信息，例如用户 ID、用户名和角色。
            不应在这里保存密码等敏感信息。
        expire: Token 有效期，单位为秒。未传入时使用配置项
            ``settings.token_max_age``。

    Returns:
        无返回值。Redis 客户端未创建时直接结束，不写入数据；Redis 服务
        连接异常时仍会抛出异常。
    """
    if not redis_module.redis_client:
        return

    key = _get_token_key(token_id)
    expire_time = expire or settings.token_max_age

    # 字典需要先转换成 JSON 字符串，才能作为普通字符串存入 Redis。
    await redis_module.redis_client.setex(key, expire_time, json.dumps(data))


async def remove_token(token_id: str):
    """从 Redis 删除指定 Token，使对应的登录状态立即失效。

    通常在用户主动退出登录、修改密码或账号被管理员强制下线时调用。
    删除后，客户端即使继续携带原来的 Token，也无法再获取用户登录信息。

    Args:
        token_id: 需要注销的 Token ID。

    Returns:
        无返回值。Token 不存在时 Redis 的删除操作也不会报错；Redis 客户端
        未创建时直接结束，Redis 服务连接异常时仍会抛出异常。
    """
    if not redis_module.redis_client:
        return

    key = _get_token_key(token_id)
    await redis_module.redis_client.delete(key)
