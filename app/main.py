"""主应用入口"""

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.controller import assistant_router, chat_router, user_router
from app.core.config import settings
from app.core.database import database
from app.core.exceptions import BusinessException, ErrorCode
from app.core.redis import redis_client
from app.schemas.common import BaseResponse

# 配置根 logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
    ],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 在这里运行整个 FastAPI 应用
    # 启动时执行
    await database.connect()
    print(f"数据库连接成功: {settings.db_host}:{settings.db_port}/{settings.db_name}")

    try:
        await redis_client.ping()
        print(
            f"Redis 连接成功: "
            f"{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        )

        # ExitStack 让 Workflow 的 Redis Saver 活得和当前 worker 一样久。
        # Saver 内部使用独立连接池，退出这个上下文时才关闭，
        # 不会在每次保存 checkpoint 后断开。
        async with AsyncExitStack() as stack:
            # 在基础设施就绪后，根据配置只创建一种多 Agent 编排入口。
            # FastAPI 只有执行到 yield 后才开始接收请求，因此所有
            # Controller 都可以复用同一个进程级 agent_runner。
            if settings.agent_mode == "subagents":
                from app.agents.subagents.master import MasterAgent

                agent_runner = MasterAgent()
            elif settings.agent_mode == "workflow":
                from langgraph.checkpoint.redis.aio import AsyncRedisSaver

                from app.agents.workflow.runner import WorkflowRunner

                checkpointer = await stack.enter_async_context(
                    AsyncRedisSaver.from_conn_string(
                        settings.redis_url,
                        ttl={
                            "default_ttl": settings.agent_checkpoint_ttl_minutes,
                            "refresh_on_read": (
                                settings.agent_checkpoint_refresh_on_read
                            ),
                        },
                    )
                )
                # from_conn_string() 的 __aenter__ 已经执行 asetup()，
                # 这里不要再重复初始化 Redis 索引。
                agent_runner = WorkflowRunner(checkpointer=checkpointer)
            else:
                raise ValueError(f"不支持的 Agent 模式: {settings.agent_mode}")

            # Controller 只依赖统一的 run_stream 接口，不需要知道底层是
            # MasterAgent 还是 WorkflowRunner。
            app.state.agent_runner = agent_runner
            initialize = getattr(agent_runner, "initialize", None)
            if initialize is not None:
                await initialize()
            print(f"Agent 编排模式已启动: {settings.agent_mode}")

            yield  # FastAPI 在这里持续处理请求
    finally:
        # 关闭时执行
        await redis_client.aclose()
        await database.disconnect()
        print("应用已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="AI PPT创作器",
    description="基于多智能体编排的 AI PPT 创作平台",
    version="0.0.1",
    lifespan=lifespan,
)

# 本地前端与 FastAPI 使用不同端口，浏览器会将请求视为跨域请求。
# 开发阶段允许 localhost / 127.0.0.1 上的任意端口访问聊天接口，
# 不开放给其他域名；以后部署时再替换成正式前端域名即可。
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# 挂载用户模块路由：/api + /user + /register（或 /login）
app.include_router(user_router, prefix="/api")
app.include_router(assistant_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


# 全局异常处理
@app.exception_handler(BusinessException)
async def business_exception_handler(
    _request: Request,
    exc: BusinessException,
):
    """业务异常处理"""
    return JSONResponse(
        status_code=200,
        content={
            "code": exc.error_code.code,
            "data": None,
            "message": exc.message,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """全局异常处理"""
    print(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=200,
        content={
            "code": ErrorCode.SYSTEM_ERROR.code,
            "data": None,
            "message": f"系统内部异常: {exc}",
        },
    )


@app.get("/", response_model=BaseResponse)
async def root():
    """根路径"""
    return BaseResponse.success(
        {
            "message": "AI PPT创作器 - Python 后端",
            "version": "0.0.1",
            "docs": "/docs",
        }
    )


@app.get("/api/hello")
async def say_hello():
    """hello"""
    return {"message": "Hello!"}
