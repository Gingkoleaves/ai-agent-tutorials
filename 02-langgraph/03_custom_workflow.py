# ============================================================
# LangGraph 实战：智能客服工单处理系统
# 运行方式：python 03_custom_workflow.py
#
# 业务流程：
#   客户消息 → 分类(技术/账单/通用/紧急) → 生成回复草稿
#   → 质量检查 → [不通过→修改→再检查] → [通过→输出最终回复]
# ============================================================

from typing import TypedDict, Literal, Annotated
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="deepseek-chat", temperature=0, base_url="https://api.deepseek.com")


# ============================================================
# 第一部分：定义状态
# ============================================================

class TicketState(TypedDict):
    """客服工单处理的状态 —— 贯穿整个流程的数据"""
    customer_message: str                              # 客户原始消息
    category: str                                       # 工单分类：技术/账单/通用/紧急
    priority: str                                       # 优先级：低/中/高/紧急
    response_draft: str                                 # 回复草稿
    quality_result: str                                 # 质检结论：通过/不通过
    quality_feedback: str                               # 质检反馈意见
    revision_count: int                                 # 修改次数
    final_response: str                                 # 最终回复
    log: Annotated[list[str], operator.add]             # 执行日志


# ============================================================
# 第二部分：定义节点
# ============================================================

def classify_node(state: TicketState) -> dict:
    """节点1：分类 —— 识别工单类型和紧急程度"""
    print(f"  🏷️  [分类] 正在分析客户消息...")

    prompt = f"""
你是一个客服工单分类专家。请分析以下客户消息，判断工单类型和优先级。

客户消息："{state['customer_message']}"

请严格按以下格式回答（只输出两行）：
类型：技术问题 / 账单问题 / 通用咨询 / 紧急投诉
优先级：低 / 中 / 高 / 紧急
"""
    result = llm.invoke(prompt)
    text = result.content

    # 解析分类结果
    category = "通用咨询"
    for cat in ["技术问题", "账单问题", "通用咨询", "紧急投诉"]:
        if cat in text:
            category = cat
            break

    priority = "中"
    for pri in ["紧急", "高", "中", "低"]:
        if pri in text:
            priority = pri
            break

    print(f"      → 类型: {category} | 优先级: {priority}")

    return {
        "category": category,
        "priority": priority,
        "log": [f"🏷️  分类完成：{category}（{priority}优先级）"],
    }


def generate_response_node(state: TicketState) -> dict:
    """节点2：生成回复草稿 —— 根据分类结果生成专业回复"""
    print(f"  ✍️  [生成] 正在撰写回复草稿...")

    # 根据不同类型，使用不同的回复模板
    template_map = {
        "技术问题": "请技术人员排查并提供分步骤的解决方案，语气专业、亲切",
        "账单问题": "请核对账单明细并解释收费规则，语气礼貌、透明",
        "通用咨询": "请简洁明了地回答客户问题，语气友好、乐于助人",
        "紧急投诉": "请首先表达歉意和重视，然后给出具体的解决时间表，语气诚恳、紧急",
    }
    guidance = template_map.get(state["category"], template_map["通用咨询"])

    prompt = f"""
你是{state['category']}领域的客服专家。{guidance}。

客户消息："{state['customer_message']}"
优先级：{state['priority']}

请生成一段150-200字的客服回复，直接输出回复内容，不要加"回复："等前缀。
"""
    result = llm.invoke(prompt)

    print(f"      → 草稿已生成（{len(result.content)} 字）")

    return {
        "response_draft": result.content,
        "log": [f"✍️  回复草稿已生成"],
    }


def quality_check_node(state: TicketState) -> dict:
    """节点3：质量检查 —— 审核回复是否达到标准"""
    attempt = state["revision_count"] + 1
    print(f"  🔍 [质检] 第 {attempt} 次审核回复质量...")

    prompt = f"""
你是一个客服质量审核员。请审核以下回复是否达到发布标准。

客户消息："{state['customer_message']}"
工单类型：{state['category']}（{state['priority']}优先级）

待审核回复：
"{state['response_draft']}"

审核标准：
1. 是否准确回应了客户的问题？
2. 语气是否专业、礼貌、符合工单类型？
3. 是否提供了具体可操作的信息？
4. 长度是否合适（150-200字）？

请严格按此格式回答（只输出两行）：
结论：通过 或 不通过
理由：（一句话说明通过或需要改进的地方）
"""
    result = llm.invoke(prompt)
    text = result.content

    verdict = "通过" if "通过" in text.split("理由")[0] and "不通过" not in text.split("理由")[0] else "不通过"
    feedback = text.split("理由：")[-1].strip() if "理由：" in text else text

    print(f"      → 结论: {verdict} | {feedback[:50]}...")

    return {
        "quality_result": verdict,
        "quality_feedback": feedback,
        "revision_count": state["revision_count"] + 1,
        "log": [f"🔍 质检第{attempt}次：{verdict}"],
    }


def revise_response_node(state: TicketState) -> dict:
    """节点4：修改回复 —— 根据质检反馈改进"""
    print(f"  🔧 [修改] 根据反馈完善回复...")

    prompt = f"""
你是客服回复优化专家。请根据审核反馈改进以下回复。

原回复："{state['response_draft']}"

改进建议：{state['quality_feedback']}

工单类型：{state['category']}

请直接输出修改后的完整回复（150-200字），不要加任何前缀。
"""
    result = llm.invoke(prompt)

    print(f"      → 修改完成（{len(result.content)} 字）")

    return {
        "response_draft": result.content,
        "log": [f"🔧 已根据反馈修改回复"],
    }


def finalize_node(state: TicketState) -> dict:
    """节点5：输出最终回复"""
    print(f"\n  ✅ [完成] 工单处理完毕！")

    print(f"\n{'─'*55}")
    print(f"📋 工单摘要：")
    print(f"   客户消息：{state['customer_message'][:60]}...")
    print(f"   类型：{state['category']} | 优先级：{state['priority']}")
    print(f"   质检次数：{state['revision_count']}")
    print(f"\n📄 最终回复：")
    print(f"{state['final_response']}")
    print('─'*55)

    return {
        "final_response": state["response_draft"],
        "log": ["✅ 工单处理完成"],
    }


# ============================================================
# 第三部分：条件路由
# ============================================================

def route_after_quality_check(state: TicketState) -> Literal["revise", "finalize"]:
    """质检后的路由：通过→输出 | 不通过且次数<3→修改 | 超过3次→强制输出"""
    if state["quality_result"] == "通过" or state["revision_count"] >= 3:
        return "finalize"
    return "revise"


# ============================================================
# 第四部分：构建图
# ============================================================

builder = StateGraph(TicketState)

# 注册节点
builder.add_node("classify", classify_node)
builder.add_node("generate_response", generate_response_node)
builder.add_node("quality_check", quality_check_node)
builder.add_node("revise_response", revise_response_node)
builder.add_node("finalize", finalize_node)

# 定义边（执行流程）
builder.set_entry_point("classify")

builder.add_edge("classify", "generate_response")          # 分类 → 生成回复
builder.add_edge("generate_response", "quality_check")     # 生成 → 质检
builder.add_conditional_edges(                              # 质检 → 条件分支
    "quality_check",
    route_after_quality_check,
    {"revise": "revise_response", "finalize": "finalize"},
)
builder.add_edge("revise_response", "quality_check")       # 修改完 → 再质检（循环）
builder.add_edge("finalize", END)                          # 完成 → 结束

graph = builder.compile()


# ============================================================
# 第五部分：运行
# ============================================================

def process_ticket(customer_message: str):
    """处理一个客服工单"""
    print(f"\n{'='*55}")
    print(f"📨 新工单：{customer_message[:50]}...")
    print('='*55)

    state = graph.invoke({
        "customer_message": customer_message,
        "category": "",
        "priority": "",
        "response_draft": "",
        "quality_result": "",
        "quality_feedback": "",
        "revision_count": 0,
        "final_response": "",
        "log": [],
    })

    print(f"\n📊 执行日志：")
    for entry in state["log"]:
        print(f"   {entry}")


if __name__ == "__main__":
    print("🏢 智能客服工单处理系统启动\n")

    # 测试1：技术问题
    process_ticket(
        "我的 App 更新后一直闪退，已经卸载重装好几次了还是不行，赶紧帮我看看！"
    )

    # 测试2：账单问题
    process_ticket(
        "你好，这个月的话费账单比上个月多了 80 块钱，我不知道为什么，能帮我查一下吗？"
    )

    # 测试3：通用咨询
    process_ticket(
        "请问你们的退货政策是什么？我买了一双鞋，穿了两次发现有质量问题。"
    )
