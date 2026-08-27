# app/core/database.py
from databases import Database
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# 同步引擎（建表 / DDL 用）
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # 取连接前先 ping，避免「MySQL has gone away」
    pool_recycle=3600,  # 连接每小时回收，防止长连接被服务端断开
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 异步数据库（FastAPI 异步查询用）
database = Database(settings.database_url.replace("+pymysql", ""))


async def get_db():
    """FastAPI 依赖注入： yield 生成器确保连接正确归还"""
    yield database
