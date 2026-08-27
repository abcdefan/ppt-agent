"""内容优化工具 - 调用 LLM 精炼幻灯片内容"""

import json
import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_REFINE_PROMPT = """你是一位专业的 PPT 内容专家。请对以下幻灯片 JSON 数据进行全面优化：

## 优化要求

1. **要点精简**：每个要点控制在 15 字以内，使用短语而非句子，用有力动词开头
2. **演讲备注**：为每页生成 "speaker_notes" 字段（2-3 句话的详细说明）
3. **结构优化**：确保每页只有一个核心主题，要点之间有逻辑递进
4. **来源保护**：如果原始 speaker_notes 包含以 "Sources:" 开头的来源区块，
   必须逐字保留其中的来源标题和 URL，并继续放在优化后备注的末尾；不得删除、
   改写或编造任何 URL

## 输出格式

输出与输入格式一致的 JSON 数组，保留原有字段并新增 speaker_notes 字段。
直接输出 JSON，不要添加任何解释文字。

原始幻灯片 JSON 数据：
{slides_json}"""


def _create_refiner_llm() -> ChatOpenAI:
    """创建内容优化专用 LLM 实例"""
    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0.3,
    )


def _parse_json(text: str):
    """从 LLM 回复中提取 JSON"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ```
    if "```json" in text:
        s = text.index("```json") + 7
        e = text.find("```", s)
        if e > s:
            try:
                return json.loads(text[s:e].strip())
            except json.JSONDecodeError:
                pass
    # 提取 [...]
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            pass
    return None


@tool
async def refine_content(slides_json: str) -> str:
    """优化PPT幻灯片内容：精简要点、生成演讲备注、优化结构。输入输出都是JSON格式的幻灯片数组。

    输入格式（与generate_ppt相同）：
    [{"title": "...", "bullets": [...], "layout_hint": "..."}]

    输出新增字段：speaker_notes（演讲者备注）

    Args:
        slides_json: JSON字符串格式的幻灯片数组
    """
    try:
        slides_data = json.loads(slides_json)
        if not isinstance(slides_data, list) or not slides_data:
            return "错误：slides_json 必须是非空的 JSON 数组"

        llm = _create_refiner_llm()
        prompt = _REFINE_PROMPT.format(slides_json=slides_json)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        refined = _parse_json(response.content)

        if refined and isinstance(refined, list):
            refined_json = json.dumps(refined, ensure_ascii=False, indent=2)
        else:
            logger.warning("LLM 内容优化结果 JSON 解析失败，返回原始数据")
            refined_json = slides_json

        result = {
            "success": True,
            "refined_slides": refined_json,
            "message": f"内容优化完成，优化了 {len(slides_data)} 页幻灯片",
        }
        logger.info("内容优化完成: %d 页", len(slides_data))
        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        return f"错误：slides_json 解析失败: {e}"
    except Exception as e:
        logger.error("内容优化失败: %s", e)
        return f"错误：内容优化失败: {e}"
