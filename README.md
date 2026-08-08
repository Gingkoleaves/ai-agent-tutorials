# AI Agent 框架从零开始

> 配套博客：[xuqi2024.github.io](https://xuqi2024.github.io)

这里是系列教程的所有可运行代码。每个目录对应一篇文章，代码经过验证，可以直接运行。

## 系列目录

| 章节 | 框架 | 核心概念 | 目录 |
|------|------|----------|------|
| 第 1 篇 | **LangChain** | Agent / Tool / Executor | [01-langchain](./01-langchain/) |
| 第 2 篇 | **LangGraph** | 节点 / 边 / 状态图 | [02-langgraph](./02-langgraph/) |
| 第 3 篇 | **CrewAI** | 角色 / 任务 / Crew | [03-crewai](./03-crewai/) |
| 第 4 篇 | **AutoGen** | 多智能体对话 | [04-autogen](./04-autogen/) |
| 第 5 篇 | **Agno** | 极简 Agent | [05-agno](./05-agno/) |
| 第 6 篇 | **Smolagents** | 轻量 HF Agent | [06-smolagents](./06-smolagents/) |

## 环境准备

所有示例都需要 OpenAI API Key（或兼容接口）：

```bash
export OPENAI_API_KEY="sk-..."
```

每个章节目录内都有 `requirements.txt`，进入对应目录后运行：

```bash
pip install -r requirements.txt
```

Python 版本要求：**3.10+**
> 本仓库实际运行环境：**Python 3.14.6**，Conda 环境 `AutoGen`

## 快速开始

```bash
git clone https://github.com/xuqi2024/ai-agent-tutorials.git
cd ai-agent-tutorials/01-langchain
pip install -r requirements.txt
python 01_hello_agent.py
```

## LangChain 1.3+ API 适配说明

本仓库代码已针对 `langchain==1.3.14` 作出修改，原始教程使用的旧版 API 已被废弃。主要变更：

| 旧 API | 新 API（LangChain 1.3+） |
|--------|--------------------------|
| `create_tool_calling_agent(llm, tools, prompt)` | `create_agent(model=llm, tools=tools, system_prompt="...")` |
| `AgentExecutor(agent=agent, tools=tools)` | 不再需要，`create_agent` 返回的 graph 可直接调用 |
| `ChatPromptTemplate.from_messages([...])` | 直接传 `system_prompt` 字符串 |
| `MessagesPlaceholder` | 不再需要，在 `messages` 列表中直接传递历史消息 |
| `result["output"]` | `result["messages"][-1].content` |

涉及文件：`01-langchain/01_hello_agent.py`、`01-langchain/02_memory_agent.py`
