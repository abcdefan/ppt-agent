# app/models/user.py
from sqlalchemy import BigInteger, Column, DateTime, Integer, SmallInteger, String
from sqlalchemy.sql import func

from app.core.database import Base


class User(Base):
    """系统用户表"""

    __tablename__ = "sys_user"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="id")
    username = Column(
        "username", String(32), nullable=False, unique=True, comment="用户名"
    )
    password = Column("password", String(64), nullable=False, comment="密码")
    nickname = Column("nickname", String(32), nullable=True, comment="昵称")
    user_role = Column(
        "userRole",
        String(64),
        nullable=False,
        default="user",
        comment="角色：user/admin",
    )
    create_time = Column(
        "createTime", DateTime, nullable=False, default=func.now(), comment="创建时间"
    )
    ppt_quota = Column(
        "ppt_quota", Integer, nullable=False, default=10, comment="PPT创作配额"
    )
    status = Column(
        "status",
        SmallInteger,
        nullable=False,
        default=1,
        comment="状态：0-禁用，1-启用",
    )
