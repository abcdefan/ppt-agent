"""聊天会话原始消息 ORM 模型。"""

from sqlalchemy import Column, ForeignKey, Index, JSON, String, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER, MEDIUMTEXT

from app.core.database import Base


class ChatMessageModel(Base):
    """一条用户或 Assistant 的最终可见消息。"""

    __tablename__ = "chat_message"
    __table_args__ = (
        Index("idx_chat_message_session_id", "session_id", "id"),
        Index("idx_chat_message_run_id", "run_id", "id"),
        {"comment": "聊天会话原始消息"},
    )

    id = Column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True,
        comment="消息自增 ID，也用于会话内排序和分页",
    )
    session_id = Column(
        String(64),
        ForeignKey(
            "chat_session.session_id",
            name="fk_chat_message_session",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="消息所属聊天会话",
    )
    run_id = Column(
        String(64),
        ForeignKey(
            "workflow_run.run_id",
            name="fk_chat_message_run",
            onupdate="CASCADE",
            ondelete="SET NULL",
        ),
        nullable=True,
        comment="关联的 Workflow Run",
    )
    message_role = Column(String(16), nullable=False, comment="USER/ASSISTANT")
    message_type = Column(
        String(32),
        nullable=False,
        server_default=text("'TEXT'"),
        comment="TEXT/WORKFLOW_RESULT/ERROR",
    )
    content = Column(MEDIUMTEXT, nullable=False, comment="消息原始文本内容")
    metadata_json = Column(JSON, nullable=True, comment="消息附加数据")
    token_count = Column(
        INTEGER(unsigned=True),
        nullable=True,
        comment="消息 Token 数量",
    )
    message_status = Column(
        String(16),
        nullable=False,
        server_default=text("'COMPLETED'"),
        comment="COMPLETED/FAILED",
    )
    created_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
    )
