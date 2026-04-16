# ============================================================
# 第 2 步：记忆（Memory）—— 让 Agent 记住对话历史
# 运行方式：python 02_memory_agent.py
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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

# MessagesPlaceholder：给历史消息留一个占位符
# variable_name="chat_history"：占位符的名字
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有记忆的天气助手，记住用户说过的信息。用中文回答。"),
    MessagesPlaceholder(variable_name="chat_history"),   # 历史消息放这里
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ============================================================
# 手动维护对话历史
# 列表（list）：用方括号 [] 表示，可以动态添加元素
# ============================================================
chat_history = []   # 空列表，存放对话历史


def chat(user_input: str):
    """多轮对话函数"""
    print(f"\n🙋 你：{user_input}")

    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history,  # 传入历史记录
    })

    answer = result["output"]
    print(f"🤖 AI：{answer}")

    # .append()：向列表末尾添加元素
    chat_history.append(HumanMessage(content=user_input))  # 存用户消息
    chat_history.append(AIMessage(content=answer))          # 存 AI 回复

    return answer


if __name__ == "__main__":
    print("💬 多轮对话演示（Ctrl+C 退出）\n")
    chat("你好！我叫小明，我在北京工作。")
    chat("我今天要去上海出差，那里天气怎么样？")
    chat("我之前提到我在哪个城市来着？")  # 测试记忆
    chat("北京和上海温度差多少？")
