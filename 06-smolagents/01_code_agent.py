# ============================================================
# 第 6 篇：Smolagents —— HuggingFace 的轻量级 Agent 框架
# 运行方式：python 01_code_agent.py
# ============================================================

# ---------- Python 基础说明 ----------
# *args 和 **kwargs：可变参数
# def func(*args):      <- 接受任意数量的位置参数，args 是元组
# def func(**kwargs):   <- 接受任意数量的关键字参数，kwargs 是字典
#
# 示例：
# func(1, 2, 3)  -> args = (1, 2, 3)
# func(a=1, b=2) -> kwargs = {"a": 1, "b": 2}
# -------------------------------------

from smolagents import (
    CodeAgent,        # CodeAgent：把 Python 代码作为 Action 的 Agent
    ToolCallingAgent, # ToolCallingAgent：传统 JSON 工具调用的 Agent
    tool,             # @tool 装饰器
    LiteLLMModel,     # 通用 LLM 接口（支持 OpenAI、Anthropic 等）
)


# ============================================================
# 第一部分：定义工具
# Smolagents 的工具定义：必须有完整的 docstring（文档字符串）
# 因为 Agent 会读取 docstring 来理解工具的用途和参数
# ============================================================

@tool
def get_word_count(text: str) -> int:
    """
    计算文本中的单词/字符数量。
    对中文文本返回字符数，对英文文本返回单词数。

    Args:
        text: 要统计的文本内容

    Returns:
        int: 字符数（中文）或单词数（英文）
    """
    # 简单判断：如果包含中文字符，按字符数统计
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if has_chinese:
        # 只统计非空白字符
        count = len([c for c in text if not c.isspace()])
        return count
    else:
        return len(text.split())


@tool
def format_as_markdown(title: str, content: str, level: int = 2) -> str:
    """
    将内容格式化为 Markdown 格式的章节。

    Args:
        title: 章节标题
        content: 章节内容
        level: 标题级别，1-6，默认为 2（## 标题）

    Returns:
        str: 格式化后的 Markdown 文本
    """
    header = "#" * min(max(level, 1), 6)   # 确保级别在 1-6 之间
    return f"{header} {title}\n\n{content}\n"


@tool
def summarize_list(items: list, max_items: int = 5) -> str:
    """
    将列表格式化为摘要字符串，超过 max_items 时截断。

    Args:
        items: 要格式化的列表
        max_items: 最多显示多少项，默认 5

    Returns:
        str: 格式化的列表字符串
    """
    display = items[:max_items]
    result = "\n".join(f"- {item}" for item in display)
    if len(items) > max_items:
        result += f"\n...以及另外 {len(items) - max_items} 项"
    return result


# ============================================================
# 第二部分：CodeAgent 演示
# CodeAgent 的特别之处：
#   - 传统 Agent：输出 JSON（{"tool": "search", "query": "..."})
#   - CodeAgent：输出可执行的 Python 代码！
#   - 代码会在沙箱中执行，结果反馈回 Agent
# ============================================================

def demo_code_agent():
    """CodeAgent 演示：Agent 通过写 Python 代码来完成任务"""
    print("\n🤖 CodeAgent 演示")
    print("="*50)
    print("特点：Agent 输出 Python 代码作为 Action，在沙箱中执行")
    print()

    # LiteLLMModel：通用 LLM 适配层
    # model_id 格式：provider/model，如 "openai/gpt-4o-mini"
    model = LiteLLMModel(model_id="openai/gpt-4o-mini", temperature=0)

    agent = CodeAgent(
        tools=[get_word_count, format_as_markdown, summarize_list],
        model=model,
        max_steps=5,              # 最多执行 5 步
        verbosity_level=2,        # 显示详细过程（0=安静, 1=基础, 2=详细）
    )

    # 任务：让 Agent 自动写代码来完成任务
    task = """
请完成以下任务：
1. 统计字符串 "人工智能正在改变世界，每一个开发者都应该了解 AI Agent 的基本原理" 的字符数
2. 创建一个包含 8 个 AI 框架名称的列表：LangChain、LangGraph、CrewAI、AutoGen、Agno、Smolagents、LlamaIndex、Haystack
3. 用 summarize_list 工具展示这个列表（最多显示5项）
4. 把结果格式化成一个 Markdown 章节，标题是"AI Agent 框架一览"
5. 输出最终的 Markdown 文本
"""

    result = agent.run(task)
    print(f"\n📄 最终结果：\n{result}")


# ============================================================
# 第三部分：ToolCallingAgent 演示（传统工具调用）
# ============================================================

def demo_tool_calling_agent():
    """ToolCallingAgent 演示：传统 JSON 格式的工具调用"""
    print("\n\n🔧 ToolCallingAgent 演示")
    print("="*50)
    print("特点：Agent 输出 JSON 格式的工具调用指令（传统方式）")
    print()

    model = LiteLLMModel(model_id="openai/gpt-4o-mini", temperature=0)

    agent = ToolCallingAgent(
        tools=[get_word_count, format_as_markdown],
        model=model,
        max_steps=3,
        verbosity_level=1,
    )

    result = agent.run(
        "统计'Smolagents是HuggingFace开发的轻量级Agent框架'的字符数，"
        "并将结果格式化为Markdown（标题：统计结果）"
    )
    print(f"\n📄 结果：\n{result}")


if __name__ == "__main__":
    print("🦤 Smolagents 入门演示")
    print("Smolagents 是 HuggingFace 出品的轻量级 Agent 框架")
    print("最大特点：CodeAgent 直接执行 Python 代码作为 Action\n")

    demo_code_agent()
    demo_tool_calling_agent()
