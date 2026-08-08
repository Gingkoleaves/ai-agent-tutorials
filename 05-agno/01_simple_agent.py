# ============================================================
# 第 5 篇：Agno —— 极简风格的多模态 Agent 框架
# 运行方式：python 01_simple_agent.py
# ============================================================

# ---------- Python 基础说明 ----------
# 装饰器（Decorator）@：用来给函数"加功能"的语法糖
# @tool
# def my_func():   <- 等价于 my_func = tool(my_func)
#    ...
#
# 类属性 vs 实例属性：
# class Dog:
#     species = "犬科"          <- 类属性（所有实例共享）
#     def __init__(self, name):
#         self.name = name      <- 实例属性（每个实例独有）
# -------------------------------------

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.storage.sqlite import SqliteStorage
import datetime


# ============================================================
# 第一部分：定义工具（使用 @tool 装饰器）
# ============================================================

@tool
def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return f"当前时间是：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。
    expression: 数学表达式字符串，如 "2 + 3 * 4"
    """
    try:
        # eval() 执行字符串表达式，这里仅允许数字和运算符
        # 注意：生产环境要做严格的输入验证
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "只支持数字和基本运算符（+、-、*、/、%）"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """
    模拟知识库搜索（真实项目中可接向量数据库）。
    query: 搜索关键词
    """
    # 简单的关键词匹配知识库
    knowledge_base = {
        "langchain": "LangChain 是一个用于构建 AI 应用的框架，特点是工具链丰富，生态完善。",
        "langgraph": "LangGraph 是 LangChain 的图编排扩展，适合需要条件分支和循环的复杂工作流。",
        "crewai": "CrewAI 专注于多 Agent 角色协作，每个 Agent 有独立的角色、目标和背景故事。",
        "autogen": "AutoGen 来自微软，擅长多 Agent 对话和代码执行，支持群组聊天模式。",
        "agno": "Agno 是一个极简、高性能的 Agent 框架，原生支持多模态、长期记忆和多 Agent。",
        "smolagents": "Smolagents 来自 HuggingFace，轻量级，支持 CodeAgent（把代码作为 Action）。",
    }

    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value

    return f"未找到关于「{query}」的具体信息，请尝试其他关键词。"


# ============================================================
# 第二部分：创建 Agent
# Agno 的 Agent 极其简洁，几行代码即可配置完成
# ============================================================

def create_agent(use_memory: bool = False):
    """创建 Agno Agent"""

    storage = None
    if use_memory:
        # SqliteStorage：用 SQLite 数据库存储对话历史（轻量级本地存储）
        # session_id：会话 ID，同一个 ID 的对话会被关联在一起
        storage = SqliteStorage(
            table_name="agent_sessions",
            db_file="agent_memory.db",
        )

    agent = Agent(
        # 模型配置
        model=OpenAIChat(id="deepseek-chat", base_url="https://api.deepseek.com"),

        # 工具列表
        tools=[get_current_time, calculate, search_knowledge],

        # 系统提示词
        instructions=[
            "你是一个有用的助手，擅长回答关于 AI Agent 框架的问题",
            "当用户询问时间时，使用 get_current_time 工具",
            "当用户需要计算时，使用 calculate 工具",
            "当用户询问 AI 框架知识时，先用 search_knowledge 工具搜索，再补充你的知识",
            "用中文回答，语言简洁友好",
        ],

        # 记忆配置
        storage=storage,
        add_history_to_messages=use_memory,   # 是否将历史消息加入上下文
        num_history_responses=5,               # 记住最近5条对话

        # 显示工具调用过程
        show_tool_calls=True,
        markdown=True,
    )

    return agent


# ============================================================
# 第三部分：运行演示
# ============================================================

def demo_basic():
    """基础功能演示"""
    print("\n🚀 Agno Agent 基础演示")
    print("="*50)

    agent = create_agent(use_memory=False)

    # agent.print_response()：调用 Agent 并打印格式化输出
    questions = [
        "现在几点了？",
        "帮我计算 (100 + 50) * 2 / 3",
        "LangGraph 和 CrewAI 有什么区别？请分别搜索它们的信息后比较。",
    ]

    for q in questions:
        print(f"\n❓ 问题：{q}")
        agent.print_response(q, stream=False)


def demo_memory():
    """记忆功能演示"""
    print("\n\n🧠 Agno 记忆功能演示（同一会话内记住上下文）")
    print("="*50)

    agent = create_agent(use_memory=True)

    # 多轮对话
    conversations = [
        "我叫小明，我在学习 AI Agent 框架",
        "我刚才说我叫什么名字？",   # 测试记忆
        "我现在最感兴趣的是 Agno，帮我搜索一下它的特点",
    ]

    for msg in conversations:
        print(f"\n👤 用户：{msg}")
        agent.print_response(msg, stream=False)


if __name__ == "__main__":
    demo_basic()
    demo_memory()
