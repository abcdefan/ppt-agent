"""Workflow 子图的公共构建入口。"""

from app.agents.workflow.subgraphs.assets import (
    build_assets_subgraph,
    build_debug_assets_subgraph,
    route_assets,
)
from app.agents.workflow.subgraphs.create import (
    DEBUG_PPT_CREATION_SUBGRAPH,
    PPT_CREATION_SUBGRAPH,
    build_debug_ppt_creation_subgraph,
    build_ppt_creation_subgraph,
    route_after_assets,
    route_after_content,
    route_create_entry,
)
from app.agents.workflow.subgraphs.edit import (
    PPT_EDIT_SUBGRAPH,
    build_ppt_edit_subgraph,
)

__all__ = [
    "DEBUG_PPT_CREATION_SUBGRAPH",
    "PPT_CREATION_SUBGRAPH",
    "PPT_EDIT_SUBGRAPH",
    "build_assets_subgraph",
    "build_debug_assets_subgraph",
    "build_debug_ppt_creation_subgraph",
    "build_ppt_creation_subgraph",
    "build_ppt_edit_subgraph",
    "route_after_assets",
    "route_after_content",
    "route_create_entry",
    "route_assets",
]
