"""文件操作工具 - 读取、写入、列出文件"""

import logging
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import settings

logger = logging.getLogger(__name__)


def _resolve_safe_path(relative_path: str) -> Path:
    """将相对路径解析为沙箱内的安全绝对路径，防止路径穿越"""
    base = settings.workspace_path
    resolved = (base / relative_path).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"路径越权访问: {relative_path}")
    return resolved


@tool
async def read_file(path: str) -> str:
    """读取文件内容。path 为相对于工作区的路径"""
    try:
        file_path = _resolve_safe_path(path)
        if not file_path.exists():
            return f"错误：文件不存在: {path}"
        if file_path.is_dir():
            return f"错误：路径是目录，不是文件: {path}"
        content = file_path.read_text(encoding="utf-8")
        logger.info("读取文件成功: %s (%d 字符)", path, len(content))
        return content
    except ValueError as e:
        return f"错误：{e}"
    except OSError as e:
        logger.exception("读取文件失败: %s", path)
        return f"错误：读取文件失败: {e}"


@tool
async def write_file(path: str, content: str) -> str:
    """将内容写入文件。path 为相对于工作区的路径。会自动创建不存在的父目录。"""
    try:
        file_path = _resolve_safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.info("写入文件成功: %s (%d 字符)", path, len(content))
        return f"文件写入成功: {path}"
    except ValueError as e:
        return f"错误：{e}"
    except OSError as e:
        logger.exception("写入文件失败: %s", path)
        return f"错误：写入文件失败: {e}"


@tool
async def list_files(directory: str = ".", pattern: str = "*") -> str:
    """列出目录中的文件。directory 为相对于工作区的路径。"""
    try:
        dir_path = _resolve_safe_path(directory)
        if not dir_path.exists():
            return f"错误：目录不存在: {directory}"
        if not dir_path.is_dir():
            return f"错误：路径不是目录: {directory}"
        files = sorted(dir_path.glob(pattern))
        if not files:
            return f"目录 {directory} 中没有匹配 '{pattern}' 的文件"
        lines = []
        for f in files:
            prefix = "[目录]" if f.is_dir() else "[文件]"
            size = f.stat().st_size if f.is_file() else "-"
            lines.append(f"{prefix} {f.relative_to(dir_path)} ({size} bytes)")
        result = "\n".join(lines)
        logger.info("列出文件: %s, 共 %d 项", directory, len(files))
        return result
    except ValueError as e:
        return f"错误：{e}"
    except OSError as e:
        logger.exception("列出文件失败: %s", directory)
        return f"错误：列出文件失败: {e}"
