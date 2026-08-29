"""PPT 当前业务记录 ORM 模型。"""

from sqlalchemy import Column, Index, JSON, String, Text, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER

from app.core.database import Base


class PptRecordModel(Base):
    """一份逻辑 PPT 的当前业务快照。"""

    __tablename__ = "ppt_record"
    __table_args__ = (
        Index(
            "uk_ppt_record_current_file_key",
            "current_file_key",
            unique=True,
        ),
        Index("idx_ppt_record_owner_updated", "owner_user_id", "updated_at"),
        Index(
            "idx_ppt_record_owner_status",
            "owner_user_id",
            "lifecycle_status",
        ),
        {"comment": "PPT 当前业务记录"},
    )

    ppt_id = Column(String(64), primary_key=True, comment="逻辑 PPT ID")
    owner_user_id = Column(
        BIGINT(),
        nullable=False,
        comment="所有者用户 ID，本地模式固定为 1",
    )
    title = Column(String(255), nullable=True, comment="PPT 标题")
    style = Column(
        String(32),
        nullable=False,
        server_default=text("'business'"),
        comment="PPT 风格",
    )
    lifecycle_status = Column(
        String(32),
        nullable=False,
        server_default=text("'PLANNING'"),
        comment="PLANNING/READY/ARCHIVED",
    )
    source_type = Column(
        String(32),
        nullable=False,
        server_default=text("'GENERATED'"),
        comment="GENERATED/UPLOADED",
    )
    current_version = Column(
        INTEGER(unsigned=True),
        nullable=False,
        server_default=text("0"),
        comment="当前文件版本",
    )
    current_filename = Column(String(255), nullable=True, comment="下载展示文件名")
    current_file_key = Column(
        String(512, collation="utf8mb4_bin"),
        nullable=True,
        comment="本地相对路径或对象存储 Key",
    )
    outline_json = Column(JSON, nullable=True, comment="当前大纲结构")
    research_report_json = Column(JSON, nullable=True, comment="当前调研报告")
    slides_manifest_json = Column(JSON, nullable=True, comment="当前页面内容清单")
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
    archived_at = Column(DATETIME(fsp=6), nullable=True)
