"""工具调用包"""

from .chart_generator import add_chart_slide, prepare_chart_operation
from .content_refiner import refine_content
from .enhance_ppt import enhance_ppt
from .file_tools import list_files, read_file, write_file
from .pexels_search import add_image_slide, prepare_image_operation
from .ppt_generator import generate_ppt
from .ppt_lock import acquire_ppt_lock, release_ppt_lock
from .tavily_search import fetch_url, web_search

# 所有工具列表，供 bind_tools() 使用
all_tools = [
    read_file,
    write_file,
    list_files,
    generate_ppt,
    refine_content,
    add_chart_slide,
    add_image_slide,
    enhance_ppt,
]
