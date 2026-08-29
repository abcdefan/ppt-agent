"""配置管理"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 定位项目根目录
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """应用配置"""

    # 服务器配置
    server_port: int = 8180
    server_host: str = "0.0.0.0"
    # 数据库配置
    db_host: str
    db_port: int = 3306
    db_name: str
    db_user: str
    db_password: str
    # redis配置
    redis_host: str
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    # 业务配置
    # 单用户本地运行模式使用的用户 ID；可通过 .env 的 LOCAL_USER_ID 覆盖。
    local_user_id: int = 1
    password_salt: str
    token_max_age: int = 2592000
    # ai配置
    deepseek_base_url: str
    deepseek_api_key: str
    deepseek_model: str
    # DashScope OpenAI 兼容 Embedding。Key 留空时 Router 自动降级到 LLM。
    dashscope_base_url: str
    dashscope_api_key: str
    dashscope_embedding_model: str
    dashscope_embedding_batch_size: int = 10
    intent_embedding_min_score: float = 0.72
    intent_embedding_margin: float = 0.08
    # Tavily 联网检索。留空时 Research 的搜索能力明确降级；已知 URL
    # 仍可由 fetch_url 使用 httpx 读取。
    tavily_api_key: str
    # 文件存储目录
    workspace_dir: str = "workspace"
    # 智能体配置
    agent_max_react_iterations: int = 10  # ReAct 最大迭代次数
    agent_max_plan_steps: int = 8  # 计划最大步骤数
    agent_enable_plan_review: bool = True  # 是否启用计划审查
    agent_multi_recursion_limit: int = 50  # 多 Agent 编排的最大图递归步数
    # 节点级重试上限：各节点最多尝试次数（含首次执行）。state.attempt_counts
    # 每次执行都 +1，未达上限时路由回指自身重试，达到上限后进入失败收尾
    # （outline/content）或降级继续（research/planner/beautify）。Assets 属于
    # Best-Effort，不参与节点级重试。可通过环境变量 AGENT_MAX_ATTEMPTS 覆盖。
    agent_max_attempts: dict[str, int] = {
        "research": 3,
        "outline": 3,
        "content": 3,
        "beautify": 3,
        "planner": 3,
    }
    # 多 Agent 编排模式：subagents 或 workflow。
    # 可在 .env 中通过 AGENT_MODE=workflow 切换。
    agent_mode: str = "workflow"
    # PPT 创建子图：standard 为生产单点写入；debug 保留 Image/Chart
    # 直接并行覆盖同一 PPT 的危险行为，只用于本地 Trace 实验。
    agent_ppt_subgraph_mode: Literal["standard", "debug"] = "standard"
    # pexels 图片搜索api
    pexels_api_key: str

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,  # 字段名大小写不敏感
        extra="ignore",  # .env 里多余的字段直接忽略，不报错
    )

    # 将方法转换为只读属性，调用时使用 settings.database_url，无需加括号
    @property
    def database_url(self) -> str:
        return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"

    @property
    def redis_url(self) -> str:
        """获取 Redis 连接 URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def workspace_path(self) -> Path:
        """获取工作区绝对路径"""
        return BASE_DIR / self.workspace_dir


# Pydantic Settings 会：
# 打开项目根目录下的 .env
# 读取其中的环境变量
# 按字段名进行匹配
# 自动转换字段类型
# 检查必填字段是否存在
# 创建 Settings 对象
settings = Settings()  # 模块级单例，全项目共享
