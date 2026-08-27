"""用户服务"""

from databases import Database
from sqlalchemy import and_, func, select

from app.constants.user import UserConstant
from app.core.exceptions import ErrorCode, throw_if, throw_if_not
from app.models.user import User
from app.schemas.user import LoginUserVO, UserLoginRequest, UserRegisterRequest
from app.utils.password import encrypt_password


class UserService:
    """用户服务"""

    def __init__(self, db: Database):
        self.db = db

    async def register(self, request: UserRegisterRequest) -> int:
        """用户注册"""
        # 校验参数
        throw_if(
            len(request.username) < 4, ErrorCode.PARAMS_ERROR, "用户名长度不能小于 4 位"
        )
        throw_if(
            len(request.password) < 8, ErrorCode.PARAMS_ERROR, "密码长度不能小于 8 位"
        )
        throw_if(
            request.password != request.check_password,
            ErrorCode.PARAMS_ERROR,
            "两次输入的密码不一致",
        )
        # 检查用户名是否已存在
        query = select(func.count(User.id)).where(
            and_(User.username == request.username, User.status == 1)
        )
        count = await self.db.fetch_val(query)
        throw_if(count > 0, ErrorCode.USER_ALREADY_EXIST, "用户名已存在")

        # 加密密码
        encrypted_password = encrypt_password(request.password)

        # 插入用户
        query = """
            INSERT INTO sys_user (username, password, nickname, userRole)
            VALUES (:username, :password, :nickname, :userRole)
        """
        user_id = await self.db.execute(
            query=query,
            values={
                "username": request.username,
                "password": encrypted_password,
                "nickname": f"用户{request.username}",
                "userRole": UserConstant.DEFAULT_ROLE,
            },
        )

        return user_id

    async def login(self, request: UserLoginRequest) -> LoginUserVO:
        """用户登录"""
        # 校验参数
        throw_if(
            len(request.username) < 4, ErrorCode.PARAMS_ERROR, "用户名长度不能小于 4 位"
        )
        throw_if(
            len(request.password) < 8, ErrorCode.PARAMS_ERROR, "密码长度不能小于 8 位"
        )

        # 查询用户
        query = select(User).where(
            and_(User.username == request.username, User.status == 1)
        )
        user = await self.db.fetch_one(query)
        throw_if_not(user, ErrorCode.USER_NOT_EXIST, "用户不存在")
        # 验证密码
        encrypted_password = encrypt_password(request.password)
        throw_if(
            user["password"] != encrypted_password, ErrorCode.PASSWORD_ERROR, "密码错误"
        )

        user_dict = dict(user)

        # 返回登录用户信息
        return LoginUserVO(
            id=user_dict["id"],
            username=user_dict["username"],
            nickname=user_dict["nickname"],
            user_role=user_dict["userRole"],
            createTime=user_dict["createTime"].isoformat(),
        )
