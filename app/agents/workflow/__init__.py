"""基于显式工作流图的 Agent 编排方式。

Node、Graph 与 State 更新分为三个阶段：

    Node 定义阶段
    → 决定当前 Node 执行后返回什么 State patch

    Graph 构建阶段
    → 通过 StateGraph(WorkflowState) 读取每个字段的合并规则

    Graph 运行阶段
    → 真正将 Node 返回的 patch 合并进 State，再传给下一个 Node

因此，Node 只负责读取当前 State 并返回局部更新；LangGraph 负责根据
WorkflowState 中的默认覆盖规则或 add_messages reducer 完成实际合并。

外层 Graph 与内部 Agent / LLM 的调用层级如下：

    graph.ainvoke(initial_state)
    │
    ├─ Research Node(state)
    │    └─ research_agent.ainvoke(user_message)
    │         └─ web_search / fetch_url Tools
    │
    ├─ Outline Node(state)
    │    └─ outline_agent.ainvoke(user_message + research_report)
    │         └─ LLM 调用
    │
    ├─ Enhancement Planner Node(state)
    │    └─ structured_llm.ainvoke(user_message + research_report + outline)
    │
    ├─ Content Node(state)
    │    └─ content_agent.ainvoke(messages)
    │         ├─ LLM 调用
    │         ├─ refine_content Tool
    │         ├─ generate_ppt Tool
    │         └─ LLM 最终回复
    │
    ├─ 可选 Assets 子图
    │    └─ Image/Chart 并行准备 → Edit 单点写入
    │
    ├─ 可选 Beautify Node(state)
    │
    └─ Finalize Create → Persist PPT Record → END

Graph 的 ainvoke() 接收并传递 WorkflowState；运行到 Agent Node 时，Node 再从
State 构造 messages，调用内部 Specialist Agent 的 ainvoke()，最后把 Agent
结果转换成 State patch 返回给外层 Graph。

当前各文件的责任与调用顺序：

    state.py
    → 定义整张图流动的 WorkflowState

    nodes/
    → 定义 Router、Reply、Specialist、PPT Context 以及 Create/Edit 专用节点

    subgraphs/
    → 在 create/、edit/、assets/ 中分别组装对应子图

    graph.py
    → 注册 Node、连接边并 compile() 成可执行 Graph

    streaming.py
    → 将 Graph astream_events() 转成项目现有业务事件

    runner.py
    → 构造 initial_state，调用 graph.ainvoke()/astream_events()，
      并统一处理会话记忆和 DONE/ERROR
"""
