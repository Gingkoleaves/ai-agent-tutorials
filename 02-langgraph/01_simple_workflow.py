# ============================================================
# 第 2 篇：LangGraph —— 用"流程图"编排 AI 工作流
# 运行方式：python 01_simple_workflow.py
# ============================================================

# ---------- Python 基础说明 ----------
# TypedDict：定义"有类型的字典"，让代码更清晰
#   普通字典：{"name": "Alice", "age": 25}（无类型限制）
#   TypedDict：像定义表格列名和类型，让 IDE 和工具知道有哪些字段
#
# Annotated：给类型加"注释"，list[str] 表示字符串列表
# operator.add：两个列表合并的函数，LangGraph 用它来合并同名字段
# -------------------------------------

from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 第一部分：定义状态（State）
# State 就是"流程中传递的数据包"，每个节点都能读取和修改它
# ============================================================

class ResearchState(TypedDict):
    """研究助手的工作状态"""
    topic: str                              # 研究主题（字符串）
    search_results: str                     # 搜索结果
    analysis: str                           # 分析内容
    final_report: str                       # 最终报告
    # Annotated[list[str], operator.add]：
    #   表示 steps 是字符串列表，每次更新是"追加"而非"替换"
    steps: Annotated[list[str], operator.add]


# ============================================================
# 第二部分：定义节点（Nodes）
# 节点 = 执行具体任务的函数
# 每个函数接收 state，返回"要更新的字段"（字典格式）
# ============================================================

def search_node(state: ResearchState) -> dict:
    """节点1：模拟搜索信息"""
    topic = state["topic"]
    print(f"  🔍 正在搜索关于「{topic}」的信息...")

    # 用 LLM 模拟搜索结果（实际可接入真实搜索 API）
    prompt = f"请用2-3句话，列举关于「{topic}」的3个关键事实。格式：1. ... 2. ... 3. ..."
    result = llm.invoke(prompt)

    return {
        "search_results": result.content,
        "steps": [f"✅ 搜索完成"],
    }


def analyze_node(state: ResearchState) -> dict:
    """节点2：分析搜索结果"""
    print(f"  🧠 正在分析搜索结果...")

    prompt = f"""
根据以下关于「{state['topic']}」的搜索结果，提炼出最重要的洞察：

搜索结果：
{state['search_results']}

请用2-3句话总结核心洞察。
"""
    result = llm.invoke(prompt)

    return {
        "analysis": result.content,
        "steps": ["✅ 分析完成"],
    }


def write_report_node(state: ResearchState) -> dict:
    """节点3：撰写最终报告"""
    print(f"  ✍️  正在撰写报告...")

    prompt = f"""
请基于以下内容，写一段简洁的研究摘要（3-4句话）：

主题：{state['topic']}
关键事实：{state['search_results']}
核心洞察：{state['analysis']}

格式：直接输出摘要，不要标题。
"""
    result = llm.invoke(prompt)

    return {
        "final_report": result.content,
        "steps": ["✅ 报告撰写完成"],
    }


# ============================================================
# 第三部分：构建图（Graph）
# 图 = 节点 + 节点之间的边（执行顺序）
# ============================================================

# StateGraph：创建一个带状态的图，括号里传入 State 类型
graph_builder = StateGraph(ResearchState)

# add_node(名字, 函数)：注册节点
graph_builder.add_node("search",       search_node)
graph_builder.add_node("analyze",      analyze_node)
graph_builder.add_node("write_report", write_report_node)

# add_edge(从哪里, 到哪里)：定义执行顺序（有向边）
graph_builder.add_edge("search",       "analyze")       # 搜索 → 分析
graph_builder.add_edge("analyze",      "write_report")  # 分析 → 写报告
graph_builder.add_edge("write_report", END)             # 写完 → 结束

# set_entry_point：设置入口节点（从哪里开始）
graph_builder.set_entry_point("search")

# compile()：编译图，生成可运行的对象
graph = graph_builder.compile()


# ============================================================
# 第四部分：运行工作流
# ============================================================

def research(topic: str):
    """运行研究工作流"""
    print(f"\n{'='*55}")
    print(f"📚 研究主题：{topic}")
    print('='*55)

    # graph.invoke()：启动图，传入初始状态
    # 初始状态只需要提供"起始数据"，其余字段为空
    initial_state = {
        "topic": topic,
        "search_results": "",
        "analysis": "",
        "final_report": "",
        "steps": [],
    }

    final_state = graph.invoke(initial_state)

    print(f"\n{'─'*55}")
    print("📋 执行步骤：")
    for step in final_state["steps"]:    # 遍历列表
        print(f"   {step}")

    print(f"\n📄 最终报告：")
    print(final_state["final_report"])
    print('='*55)


if __name__ == "__main__":
    research("大型语言模型的工作原理")
    research("Python 异步编程")
