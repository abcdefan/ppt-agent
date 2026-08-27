"""工具子集分组 — 按角色划分工具，供两种多智能体模式共享。"""

import logging

from app.tools import (
    add_chart_slide,
    add_image_slide,
    enhance_ppt,
    fetch_page,
    generate_ppt,
    list_files,
    read_file,
    refine_content,
    web_search,
)

logger = logging.getLogger(__name__)

# 6 个 specialist 角色名（两种模式都用）；顺序即推荐流水线顺序：
# outline → research（联网检索）→ content → image → chart → beautify
ROLES = ("outline", "research", "content", "image", "chart", "beautify")


def split_tools() -> dict[str, list]:
    """按角色划分工具子集。read_file 在多组共享（无副作用，可重复引用）。"""
    return {
        "outline": [read_file, list_files],
        "research": [web_search, fetch_page, read_file, list_files],
        "content": [refine_content, generate_ppt, read_file, list_files],
        "image": [add_image_slide, read_file],
        "chart": [add_chart_slide, read_file],
        "beautify": [enhance_ppt, read_file],
    }
