"""智能体模块"""

from app.agent.langchain_agent import LangChainAgent
from app.agent.ppt_agent import PPTAgent

# 手动实现的智能体（ReAct + Plan-Execute）
ppt_agent = PPTAgent()

# LangChain 智能体
langchain_agent = LangChainAgent()

__all__ = ["langchain_agent", "ppt_agent"]
