"""Tavily 联网搜索工具，供 Research Specialist 检索和读取网页。"""

import json
import logging
import re

import httpx
from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
SNIPPET_MAX_CHARS = 500
FETCH_MAX_CHARS = 4_000


def _client() -> AsyncTavilyClient:
    """按调用创建一个异步 Tavily Client。"""
    return AsyncTavilyClient(api_key=settings.tavily_api_key)


@tool
async def web_search(query: str) -> str:
    """使用 Tavily 搜索网页，返回标题、URL 和内容摘要。"""
    logger.info("[tavily/web_search] query=%s", query[:80])

    if not settings.tavily_api_key:
        return json.dumps(
            {
                "success": False,
                "query": query,
                "results": [],
                "message": "未配置 TAVILY_API_KEY，跳过联网搜索",
            },
            ensure_ascii=False,
        )

    try:
        response = await _client().search(
            query=query,
            max_results=MAX_RESULTS,
            include_raw_content=False,
        )
        results = []
        for item in (response or {}).get("results", []) or []:
            content = str(item.get("content") or "").strip()
            if len(content) > SNIPPET_MAX_CHARS:
                content = content[:SNIPPET_MAX_CHARS] + "…"
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": content,
                }
            )

        logger.info("[tavily/web_search] 返回 %d 条结果", len(results))
        return json.dumps(
            {
                "success": True,
                "query": query,
                "results": results,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.warning("[tavily/web_search] 搜索失败: %s", exc)
        return json.dumps(
            {
                "success": False,
                "query": query,
                "results": [],
                "message": f"搜索失败: {exc}",
            },
            ensure_ascii=False,
        )


@tool
async def fetch_url(url: str) -> str:
    """提取单个网页正文；Tavily Extract 失败时回退到 httpx。"""
    logger.info("[tavily/fetch_url] url=%s", url[:100])

    if not url or not url.lower().startswith(("http://", "https://")):
        return json.dumps(
            {
                "success": False,
                "url": url,
                "content": "",
                "message": "无效的 URL",
            },
            ensure_ascii=False,
        )

    if settings.tavily_api_key:
        try:
            response = await _client().extract([url])
            results = (response or {}).get("results", []) or []
            if results:
                content = str(
                    results[0].get("raw_content")
                    or results[0].get("content")
                    or ""
                ).strip()
                if content:
                    return json.dumps(
                        {
                            "success": True,
                            "url": url,
                            "content": content[:FETCH_MAX_CHARS],
                        },
                        ensure_ascii=False,
                    )
        except Exception as exc:
            logger.warning("[tavily/fetch_url] Extract 失败，回退 httpx: %s", exc)

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 PPTCreatorBot"},
            )
            response.raise_for_status()
            content = response.text
    except Exception as exc:
        logger.warning("[tavily/fetch_url] httpx 抓取失败: %s", exc)
        return json.dumps(
            {
                "success": False,
                "url": url,
                "content": "",
                "message": f"抓取失败: {exc}",
            },
            ensure_ascii=False,
        )

    content = re.sub(
        r"<(script|style|noscript)[^>]*>.*?</\1>",
        " ",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"\s+", " ", content).strip()

    return json.dumps(
        {
            "success": True,
            "url": url,
            "content": content[:FETCH_MAX_CHARS],
        },
        ensure_ascii=False,
    )
