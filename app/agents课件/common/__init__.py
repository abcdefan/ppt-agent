"""多智能体共享基础设施 — 供 workflow / subagents 两种模式复用。

- llm.py     LLM 工厂（DashScope ChatOpenAI）
- tools.py   工具子集分组（split_tools + ROLES）
- prompts.py 4 个 specialist 角色提示词（ROLE_PROMPTS）
"""
