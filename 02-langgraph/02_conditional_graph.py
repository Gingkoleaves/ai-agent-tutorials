# ============================================================
# LangGraph 进阶：条件分支 —— Agent 根据情况走不同的路
# 运行方式：python 02_conditional_graph.py
# ============================================================

from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ReviewState(TypedDict):
    content: str        # 待审核内容
    verdict: str        # 审核结论："通过" 或 "不通过"
    feedback: str       # 反馈意见
    revision: str       # 修改后的内容
    attempts: int       # 修改次数


def review_node(state: ReviewState) -> dict:
    """节点：审核内容质量"""
    print(f"  🔍 审核中... (第 {state['attempts'] + 1} 次)")

    prompt = f"""
请审核以下内容，判断它是否足够专业和完整（至少80字，包含具体数据或例子）：

内容：{state['content'] if state['attempts'] == 0 else state['revision']}

回答格式（严格按此格式）：
结论：通过 或 不通过
理由：一句话说明原因
"""
    result = llm.invoke(prompt)
    text = result.content

    # 解析结论
    verdict = "通过" if "通过" in text.split("理由")[0] and "不通过" not in text.split("理由")[0] else "不通过"
    feedback = text.split("理由：")[-1].strip() if "理由：" in text else text

    print(f"  📋 审核结论：{verdict}")
    return {"verdict": verdict, "feedback": feedback, "attempts": state["attempts"] + 1}


def revise_node(state: ReviewState) -> dict:
    """节点：根据反馈修改内容"""
    print(f"  ✏️  根据反馈修改内容...")
    base = state["revision"] if state["revision"] else state["content"]

    prompt = f"""
请改进以下内容，使其更专业完整（至少100字，加入具体数据或案例）：

原内容：{base}
改进建议：{state['feedback']}

直接输出改进后的内容，不要任何前缀。
"""
    result = llm.invoke(prompt)
    return {"revision": result.content}


def accept_node(state: ReviewState) -> dict:
    """节点：内容通过审核"""
    final = state["revision"] if state["revision"] else state["content"]
    print(f"\n  ✅ 内容审核通过！")
    print(f"\n📄 最终内容：\n{final}")
    return {}


# ============================================================
# 条件路由函数：根据状态决定走哪条边
# Literal["revise", "accept"]：返回值只能是这两个字符串之一
# ============================================================

def route_after_review(state: ReviewState) -> Literal["revise", "accept"]:
    """审核后的路由：通过→接受，不通过且次数<3→修改，超过3次→强制接受"""
    if state["verdict"] == "通过" or state["attempts"] >= 3:
        return "accept"
    return "revise"


# 构建图
builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_node("revise", revise_node)
builder.add_node("accept", accept_node)

builder.set_entry_point("review")

# add_conditional_edges：根据函数返回值决定走哪条边
# 格式：add_conditional_edges(起始节点, 路由函数, {返回值: 目标节点})
builder.add_conditional_edges(
    "review",
    route_after_review,
    {"revise": "revise", "accept": "accept"}
)
builder.add_edge("revise", "review")   # 修改完再审核（形成循环）
builder.add_edge("accept", END)

graph = builder.compile()


if __name__ == "__main__":
    print("📝 内容自动审核与修改系统\n")
    initial = {
        "content": "LLM 是很厉害的 AI。",   # 初始内容很简短，应该不通过
        "verdict": "",
        "feedback": "",
        "revision": "",
        "attempts": 0,
    }
    graph.invoke(initial)
