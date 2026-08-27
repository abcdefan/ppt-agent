"""Subagents-as-Tools 编排方式的完整流程说明。

这个包实现的是“Master Agent 把 Specialist Agents 当作高级 Tool 调用”的
多 Agent 架构。这里的 ``Subagent`` 描述专家相对于 Master 的调用关系；真正
负责大纲、调研、内容、配图、图表和美化工作的对象，统一称为 Specialist Agent。

本文件只用于解释模块关系和运行流程，不参与对象创建。真正的初始化入口在
``app.main.lifespan``，Master 的实现位于 ``master.py``，专家的 Tool 包装位于
``delegation_tools.py``，流式事件转换位于 ``streaming.py``。


一、从前端请求到 Master Agent
=============================

用户在前端提交内容后，浏览器会向 FastAPI Controller 发起 POST 请求。具体
Controller 会根据接口类型，从 ``request.app.state`` 取得同一个 MasterAgent：

    agent_runner = request.app.state.agent_runner

然后调用它的某个公开方法：

- 普通非流式对话调用 ``agent_runner.run(...)``；
- 流式对话调用 ``agent_runner.run_stream(...)``；
- 普通对话与创建 PPT 都调用 ``agent_runner.run_stream(...)``，由入口 Router 分流。

因此 Controller 不负责创建 Agent，也不负责判断应该调用哪个专家。它只负责
接收 HTTP 请求、整理入参、调用 MasterAgent，并将结果转换成 HTTP/SSE 响应。


二、MasterAgent 和 Specialist Agents 何时创建
============================================

FastAPI 启动时会进入 ``app.main.lifespan``。数据库和 Redis 就绪后，
如果 ``settings.agent_mode == "subagents"``，则执行：

    app.state.agent_runner = MasterAgent()

这会进入 ``MasterAgent.__init__()``，完成整支多 Agent 团队的组装：

    FastAPI lifespan
        ↓
    MasterAgent.__init__()
        ↓
    build_llm()
        创建 Master 和各 Specialist 共用的 ChatModel 对象
        ↓
    build_delegation_tools(self.llm)
        遍历 outline/research/content/image/chart/beautify 全部角色
        ↓
    build_delegation_tool(agent_role, llm)
        为当前角色调用 build_specialist_agent(agent_role, llm)
        ↓
    build_outline_agent() / build_content_agent() / ...
        创建真正负责业务的 Specialist Agent
        ↓
    delegate_to_specialist() 闭包
        持有对应 Specialist Agent，并被 @tool 包装成 Delegation Tool
        ↓
    create_agent(model=self.llm, tools=self.delegation_tools, ...)
        创建最终的 Master Agent
        ↓
    app.state.agent_runner 保存整个团队
        ↓
    lifespan 执行到 yield，FastAPI 才开始接收请求

需要准确区分：不是 ``@tool`` 或“工具注册机制”自动创建 Specialist Agent。
真正的创建动作是 ``build_delegation_tool()`` 显式调用：

    specialist_agent = build_specialist_agent(...)

``@tool`` 的职责只是把 ``delegate_to_specialist()`` 包装成 Master 能识别的
LangChain Tool。Tool 闭包一直引用 ``specialist_agent``，所以函数返回后专家
对象也不会被释放。

这些 Specialist Agents 和 Master 都在应用启动阶段创建一次，后续请求复用
这些对象，不会在每次 HTTP 请求中重新创建。每次请求自己的消息和执行状态则
由本次 ``ainvoke``/``astream_events`` 隔离，并通过 ``session_id`` 加载相应的
外部对话记忆。


三、运行时 Master 如何选择 Specialist Agent
===========================================

以 ``run_stream()`` 为例，MasterAgent 首先通过 ``session_id`` 加载历史记忆，
再把当前用户输入包装成 HumanMessage，交给已经创建好的 Master Agent。

Master LLM 能看到的主要内容包括：

- Master 的 SYSTEM_PROMPT；
- 每个 Delegation Tool 的名称、description 和参数结构；
- 当前用户消息；
- 已加载的历史消息；
- 本次执行过程中已经返回的 Tool Results。

Master 的 SYSTEM_PROMPT 和 Tool description 会告诉模型每位专家的能力与调用
条件，但它们只是给 LLM 的决策指导，并不是代码中写死的确定性路由。最终由
Master LLM 生成 Tool Call，例如：

    {
        "name": "outline_agent_tool",
        "args": {
            "task": "请为 AI 行业趋势规划一份 6 页、商务风格的 PPT 大纲"
        }
    }

其中 ``task`` 是 Master LLM 根据用户需求、历史上下文和已有 Tool Result
现场组织出的字符串，不是 Controller 预先构造的固定对象，也没有强制的业务
JSON Schema；当前 Tool Schema 只要求它是一个字符串。


四、delegate_to_specialist() 如何完成任务委派
============================================

当 Master 生成某个 ``*_agent_tool`` Tool Call 后，LangChain 根据 Tool 名称找到
对应的 BaseTool，再执行它内部包装的 ``delegate_to_specialist(task)``。

Controller 和 MasterAgent 的 ``run_stream()`` 都不会直接调用这个内部函数；
它是 LangChain 在执行 Master 的 Tool Call 时自动触发的 Tool 实现。

``delegate_to_specialist()`` 的核心逻辑是：

    result = await specialist_agent.ainvoke(
        {"messages": [HumanMessage(content=task)]}
    )

这相当于 Master 代替真实用户，向当前 Specialist Agent 发起一次新的提问：

    真实用户
        ↓ HumanMessage(user_message)
    Master Agent
        ↓ 阅读需求并生成 Tool Call 的 task
    Delegation Tool
        ↓ HumanMessage(task)
    Specialist Agent

Specialist Agent 不会自动继承 Master 的完整消息历史。它实际获得的是：

1. 自己创建时配置的 Specialist SYSTEM_PROMPT；
2. Master 本次生成并包装成 HumanMessage 的 ``task``；
3. 自己运行期间调用业务 Tool 后产生的消息。

所以调用 Content Specialist 时，Master 必须把完整大纲写进 ``task``；调用
Image、Chart 或 Beautify Specialist 时，Master 必须把真实 PPT 文件名写进
``task``。这些信息目前通过自然语言交接，而不是共享的结构化 Workflow State。


五、Specialist 的结果如何返回 Master，并触发下一位专家
=====================================================

Specialist Agent 内部可能继续调用自己拥有的底层业务 Tools，例如：

- Content Specialist 调用 ``refine_content``、``generate_ppt``；
- Image Specialist 调用 ``add_image_slide``；
- Chart Specialist 调用 ``add_chart_slide``；
- Beautify Specialist 调用 ``enhance_ppt``。

Specialist 执行完成后，``_extract_final_text()`` 从它的运行结果中提取最后一条
有效 AI 文本，作为 Delegation Tool 的字符串返回值：

    Specialist Agent 最终 AI 文本
        ↓
    delegate_to_specialist() 返回 str
        ↓
    LangChain 将该返回值变成 Master 消息状态中的 ToolMessage
        ↓
    Master LLM 再次推理

Master 再次推理时能够看到这个 ToolMessage，因此可以：

- 调用下一位 Specialist；
- 根据失败结果调整任务或停止；
- 所有工作完成后，不再调用 Tool，直接生成最终用户回复。

例如完整 PPT 的典型循环是：

    用户请求
        ↓
    Master 调用 outline_agent_tool(task=主题、页数、风格...)
        ↓
    Outline Specialist 返回大纲文本
        ↓ ToolMessage 回到 Master
    Master 调用 content_agent_tool(task=用户要求 + 完整大纲...)
        ↓
    Content Specialist 生成 PPT，并返回真实文件名
        ↓ ToolMessage 回到 Master
    Master 按需调用 image_agent_tool / chart_agent_tool
        ↓ 每次结果都以 ToolMessage 回到 Master
    Master 调用 beautify_agent_tool(task=真实文件名 + 风格...)
        ↓ ToolMessage 回到 Master
    Master 不再调用 Tool，生成最终回复

这不是多个 HTTP 请求串联，而是一次 Master Agent 执行中的多轮
“LLM → Tool → ToolMessage → LLM”循环。LangChain 的 ``create_agent`` 负责
维护这次运行内部的循环，``recursion_limit`` 用于防止无限调用。


六、流式模式额外做了什么
========================

``run_stream()`` 的多 Agent 决策过程与 ``run()`` 相同，
区别是它们使用 ``astream_events()`` 订阅整个嵌套执行过程。

``streaming.py`` 将 LangChain 原始事件转换为前端认识的业务事件：

- Master 调用 Delegation Tool → ``AGENT_SWITCH``；
- Specialist 调用底层业务 Tool → ``TOOL_CALL``；
- Tool 执行完成 → ``TOOL_RESULT``；
- Master 的最终文本 token → ``TEXT_DELTA``；
- 整次运行结束 → ``DONE``。

Specialist 的中间 ReAct 文本不会作为最终回复直接推给用户。最终文字主要来自
Master，这样前端展示的是业务过程事件和统一总结，而不是多个 Agent 的内部
推理文本混在一起。


七、面试时可以使用的简短总结
============================

项目采用 Subagents-as-Tools 多 Agent 架构。FastAPI 在 lifespan 启动阶段创建
一个进程级 MasterAgent；MasterAgent 初始化时创建共享 LLM，通过 Agent
Registry 创建全部 Specialist Agents，再利用闭包和 ``@tool`` 将每个专家包装
成 Delegation Tool，最后把这些 Tools 绑定给 Master Agent。

请求到达后，Controller 只调用 MasterAgent 的 ``run`` 或 ``run_stream`` 等统一
入口。Master LLM 根据系统提示、Tool 描述、用户上下文和已有 Tool Results 选择
专家，并生成字符串 ``task``。LangChain 执行对应 Delegation Tool，Tool 将
``task`` 包装成 HumanMessage 调用 Specialist Agent；专家结果再作为 ToolMessage
返回 Master。Master 可以继续调用下一位专家，或者在任务完成后生成最终回复。
整个协作是在一次 Agent 执行中的多轮 Tool Calling 循环，不是 Controller 手动
串联各个专家。
"""
