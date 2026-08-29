"""Workflow 单次执行记录 ORM 模型。"""

from sqlalchemy import JSON, Column, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER

from app.core.database import Base


class WorkflowRunModel(Base):
    """一次 Create/Edit Workflow 的业务执行记录。"""

    __tablename__ = "workflow_run"
    __table_args__ = (
        Index(
            "uk_workflow_run_checkpoint_thread",
            "checkpoint_thread_id",
            unique=True,
        ),
        Index("idx_workflow_run_user_updated", "user_id", "updated_at"),
        Index(
            "idx_workflow_run_user_status_updated",
            "user_id",
            "run_status",
            "updated_at",
        ),
        Index("idx_workflow_run_session_created", "session_id", "created_at"),
        Index("idx_workflow_run_ppt_created", "ppt_id", "created_at"),
        Index("idx_workflow_run_status_updated", "run_status", "updated_at"),
        {"comment": "Workflow 单次执行及恢复记录"},
    )

    run_id = Column(String(64), primary_key=True, comment="一次 Workflow 执行 ID")
    user_id = Column(BIGINT(), nullable=False, comment="发起用户 ID")
    session_id = Column(
        String(64),
        ForeignKey(
            "chat_session.session_id",
            name="fk_workflow_run_session",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment="发起执行的 Session ID",
    )
    ppt_id = Column(
        String(64),
        ForeignKey(
            "ppt_record.ppt_id",
            name="fk_workflow_run_ppt",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        nullable=True,
        comment="本次执行操作的 PPT ID",
    )
    intent = Column(String(16), nullable=False, comment="CREATE/EDIT")
    run_status = Column(
        String(32),
        nullable=False,
        server_default=text("'CREATED'"),
        comment="CREATED/RUNNING/WAITING_INPUT/SUCCEEDED/FAILED/CANCELLED",
    )
    current_stage = Column(String(64), nullable=True, comment="当前业务阶段")
    completed_stages_json = Column(JSON, nullable=False, comment="已完成阶段数组")
    required_stages_json = Column(JSON, nullable=False, comment="要求完成阶段数组")
    checkpoint_thread_id = Column(
        String(128),
        nullable=False,
        comment="LangGraph checkpoint thread_id",
    )
    graph_version = Column(
        String(64),
        nullable=False,
        server_default=text("'v1'"),
        comment="Workflow 图版本",
    )
    input_payload_json = Column(JSON, nullable=False, comment="原始执行输入")
    waiting_type = Column(String(64), nullable=True, comment="等待输入类型")
    waiting_payload_json = Column(JSON, nullable=True, comment="等待输入数据")
    error_code = Column(String(64), nullable=True, comment="失败错误码")
    error_message = Column(Text, nullable=True, comment="失败原因")
    revision = Column(
        INTEGER(unsigned=True),
        nullable=False,
        server_default=text("0"),
        comment="乐观锁版本",
    )
    created_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
    )
    started_at = Column(DATETIME(fsp=6), nullable=True, comment="首次开始时间")
    waiting_since = Column(DATETIME(fsp=6), nullable=True, comment="进入等待时间")
    finished_at = Column(DATETIME(fsp=6), nullable=True, comment="进入终态时间")
    updated_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)"),
    )
