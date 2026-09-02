"""SSE（Server-Sent Events）"""

import json

# SSE 响应头：禁用缓存与代理缓冲，保证事件实时下发
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # nginx 等代理不要缓冲
    "Connection": "keep-alive",
}


def sse_event(event_type: str, data: dict) -> str:
    """组装单条 SSE 消息

    输出：`data: {"type": <event_type>, ...}\n\n`
    """
    # 业务 data 可能包含自己的类型字段（例如 HITL input_type）。
    # SSE 顶层 type 始终以事件名为准，避免 data 覆盖后前端无法分发。
    payload = {**data, "type": event_type}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
