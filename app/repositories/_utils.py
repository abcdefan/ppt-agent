"""Repository 内部通用转换。"""

from typing import Any


def row_to_dict(row: Any) -> dict[str, Any] | None:
    """将 databases.Record、SQLAlchemy Row 或普通 Mapping 转成 dict。"""
    if row is None:
        return None
    mapping = getattr(row, "_mapping", row)
    return dict(mapping)
