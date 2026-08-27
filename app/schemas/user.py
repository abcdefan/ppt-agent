"""用户相关请求/响应模型"""

from pydantic import BaseModel, Field


class UserRegisterRequest(BaseModel):
    """用户注册请求"""

    username: str = Field(..., min_length=4, max_length=32, description="用户名")
    password: str = Field(..., min_length=8, max_length=64, description="密码")
    check_password: str = Field(
        ..., min_length=8, max_length=512, alias="checkPassword", description="确认密码"
    )


class UserLoginRequest(BaseModel):
    """用户登录请求"""

    username: str = Field(..., min_length=4, max_length=32, description="用户名")
    password: str = Field(..., min_length=8, max_length=64, description="密码")


class UserVO(BaseModel):
    """用户视图对象"""

    id: int
    username: str
    nickname: str | None = None
    user_role: str = Field(..., alias="userRole")
    create_time: str = Field(..., alias="createTime")

    class Config:
        populate_by_name = True


class LoginUserVO(BaseModel):
    """登录用户视图对象"""

    id: int
    username: str
    nickname: str | None = None
    user_role: str = Field(..., alias="userRole")
    create_time: str = Field(..., alias="createTime")
    token: str | None = None

    class Config:
        populate_by_name = True
