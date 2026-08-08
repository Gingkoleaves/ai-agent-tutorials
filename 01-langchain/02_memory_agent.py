# ============================================================
# 第 2 步：记忆（Memory）—— 让 Agent 记住对话历史
# 运行方式：python 02_memory_agent.py
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent        # LangChain 1.3+ 新版 API
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="deepseek-chat", temperature=0, base_url="https://api.deepseek.com")


@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。"""
    weather_data = {
        "北京": "☀️ 晴天，25°C",
        "上海": "🌤️ 多云，22°C",
        "广州": "🌧️ 小雨，28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


tools = [get_weather]

# ============================================================
# 新 API：create_agent 直接传入 system_prompt 字符串
# 不需要 ChatPromptTemplate 和 MessagesPlaceholder
# ============================================================
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个有记忆的天气助手，记住用户说过的信息。用中文回答。",
)

# ============================================================
# 手动维护对话历史（多轮对话的关键）
# ============================================================
chat_history = []   # 空列表，存放所有消息


def chat(user_input: str):
    """多轮对话函数 —— 每次把完整历史 + 新问题一起传给 Agent"""
    global chat_history

    print(f"\n🙋 你：{user_input}")

    # 新 API：messages 是完整对话历史 + 当前用户消息
    result = agent.invoke({
        "messages": chat_history + [HumanMessage(content=user_input)],
    })

    # result["messages"] 包含所有消息，最后一条是 AI 的回答
    all_messages = result["messages"]
    answer = all_messages[-1].content
    print(f"🤖 AI：{answer}")

    # 把本轮对话追加到历史里，供下一轮使用
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))

    return answer


if __name__ == "__main__":
    print("💬 多轮对话演示（Ctrl+C 退出）\n")
    chat("你好！我叫小明，我在北京工作。")
    chat("我今天要去上海出差，那里天气怎么样？")
    chat("我之前提到我在哪个城市来着？")  # 测试记忆
    chat("北京和上海温度差多少？")
