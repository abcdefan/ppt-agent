"""workflow 模式 — Custom workflow（StateGraph + Supervisor 条件边路由）。

结构：
- state.py     AgentState（图共享状态）
- prompts.py   Supervisor 提示词
- agents.py    supervisor 路由节点 + specialist ReAct 节点
- graph.py     StateGraph 编排
- streaming.py astream_events → SSE 事件映射
- agent.py     入口类 MultiAgent（run/run_stream/run_ppt_stream/clear_history）

共享代码（llm/tools/specialist 提示词）在 app.agents.common。
"""
from app.agents.workflow.agent import MultiAgent

__all__ = ["MultiAgent"]
