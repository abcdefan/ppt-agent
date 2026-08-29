"""用户聊天会话 ORM 模型。"""

from sqlalchemy import Column, ForeignKey, Index, String, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME

from app.core.database import Base


class ChatSessionModel(Base):
    """一段用户对话的持久化元数据。"""

    __tablename__ = "chat_session"
    __table_args__ = (
        Index("idx_chat_session_user_updated", "user_id", "updated_at"),
        Index(
            "idx_chat_session_user_status_updated",
            "user_id",
            "lifecycle_status",
            "updated_at",
        ),
        Index("idx_chat_session_active_ppt", "active_ppt_id"),
        {"comment": "用户聊天会话"},
    )

    session_id = Column(String(64), primary_key=True, comment="聊天会话 ID")
    user_id = Column(
        BIGINT(),
        nullable=False,
        comment="所属用户 ID，本地模式固定为 1",
    )
    title = Column(
        String(255),
        nullable=False,
        server_default=text("'新对话'"),
        comment="会话标题",
    )
    lifecycle_status = Column(
        String(32),
        nullable=False,
        server_default=text("'ACTIVE'"),
        comment="ACTIVE/ARCHIVED",
    )
    active_ppt_id = Column(
        String(64),
        ForeignKey(
            "ppt_record.ppt_id",
            name="fk_chat_session_active_ppt",
            onupdate="CASCADE",
            ondelete="SET NULL",
        ),
        nullable=True,
        comment="当前会话正在讨论的 PPT ID",
    )
    created_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
    )
    updated_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text(
            "CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)"
        ),
    )
    archived_at = Column(DATETIME(fsp=6), nullable=True, comment="归档时间")
