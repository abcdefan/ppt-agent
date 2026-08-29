"""Session 与 PPT 引用关系 ORM 模型。"""

from sqlalchemy import Column, ForeignKey, Index, String, text
from sqlalchemy.dialects.mysql import DATETIME

from app.core.database import Base


class SessionPptModel(Base):
    """一段 Session 对一份 PPT 的引用。"""

    __tablename__ = "session_ppt"
    __table_args__ = (
        Index("idx_session_ppt_session_used", "session_id", "last_used_at"),
        Index("idx_session_ppt_ppt_used", "ppt_id", "last_used_at"),
        {"comment": "Session 与 PPT 的引用关系"},
    )

    session_id = Column(
        String(64),
        ForeignKey(
            "chat_session.session_id",
            name="fk_session_ppt_session",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
        comment="聊天会话 ID",
    )
    ppt_id = Column(
        String(64),
        ForeignKey(
            "ppt_record.ppt_id",
            name="fk_session_ppt_ppt",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
        comment="PPT ID",
    )
    association_source = Column(
        String(32),
        nullable=False,
        server_default=text("'SELECTED'"),
        comment="CREATED/SELECTED/UPLOADED",
    )
    first_linked_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
        comment="首次关联时间",
    )
    last_used_at = Column(
        DATETIME(fsp=6),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6)"),
        comment="最后使用时间",
    )
