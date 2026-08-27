"""用户接口"""

from typing import Annotated

from databases import Database
from fastapi import APIRouter, Depends, Response

from app.core.config import settings
from app.core.database import get_db
from app.schemas.common import BaseResponse
from app.schemas.user import LoginUserVO, UserLoginRequest, UserRegisterRequest
from app.services.user_service import UserService
from app.utils.token import generate_token_id, set_token

router = APIRouter(prefix="/user", tags=["用户管理"])


@router.post("/register", response_model=BaseResponse[int])
async def register(
    request: UserRegisterRequest,
    # FastAPI 会调用 get_db，并把它提供的数据库连接自动传给 db。
    # 等价的旧写法：db: Database = Depends(get_db)
    db: Annotated[Database, Depends(get_db)],
):
    """用户注册"""
    # UserService 依赖路由注入的 db，因此按请求实例化：方便测试时替换数据库，
    # 也能在以后改用请求级 Session 时隔离事务和状态。它本身很轻量，创建成本低。
    service = UserService(db)
    # UserService.register() 注册成功后返回数据库中新用户的 ID，类型为 int。
    user_id = await service.register(request)
    return BaseResponse.success(data=user_id, message="注册成功")


@router.post("/login", response_model=BaseResponse[LoginUserVO])
async def login(
    request: UserLoginRequest,
    response: Response,
    # Annotated 同时说明 db 的类型，以及它由 FastAPI 的依赖注入提供。
    # 等价的旧写法：db: Database = Depends(get_db)
    db: Annotated[Database, Depends(get_db)],
):
    """用户登录"""
    # 与注册接口相同，UserService 绑定路由注入的 db，按请求创建而非全局共享。
    service = UserService(db)
    # UserService.login() 返回 LoginUserVO，包含用户基本信息但不包含密码；
    # 此时 token 还是默认值 None，Controller 会在下方生成并赋值。
    user = await service.login(request)

    # 生成 Token
    token_id = generate_token_id()

    # 保存到 Redis
    await set_token(token_id, {"user": user.model_dump(by_alias=True)})

    # 设置 Cookie
    response.set_cookie(
        key="TOKEN",
        value=token_id,
        max_age=settings.token_max_age,
        httponly=True,
        samesite="lax",
    )

    user.token = token_id

    return BaseResponse.success(data=user, message="登录成功")
