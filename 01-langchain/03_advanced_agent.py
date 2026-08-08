# ============================================================
# 第 3 步：进阶 Agent —— 真实 API + 自定义工具 + 推理日志
# 运行方式：python 03_advanced_agent.py
# ============================================================

import requests

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.prebuilt.tool_node import ToolCallRequest


# ============================================================
# 扩展知识：自定义 Middleware（中间件）
# Middleware 可以拦截 Agent 的每一步执行，就像"摄像头"
# 记录 Agent 在做什么、在想什么
# ============================================================

class LoggingMiddleware(AgentMiddleware):
    """给 Agent 的每一步加上中文日志，方便观察推理过程"""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """LLM 被调用之前触发 —— Agent 准备开始新一轮思考"""
        msg_count = len(state["messages"])
        print(f"\n{'─'*40}")
        print(f"🤔 [思考轮次] 当前上下文共 {msg_count} 条消息，准备调用 LLM...")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """LLM 返回之后触发 —— 看 LLM 决定做什么"""
        last_msg = state["messages"][-1]
        # 判断 LLM 是决定调工具，还是直接给出了最终答案
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"🎯 [决策] LLM 决定调用工具: {tc['name']}({tc.get('args', {})})")
        else:
            preview = str(last_msg.content)[:100]
            print(f"💬 [回答] LLM 直接回复: {preview}...")
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """工具被调用时触发 —— 拦截并记录工具执行"""
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})
        print(f"🔧 [执行] 调用工具 {tool_name}({tool_args})")
        result = handler(request)  # 真正执行工具
        # ToolMessage 的 content 就是工具返回值
        print(f"✅ [结果] {tool_name} 返回: {str(result.content)[:120]}")
        return result


# ============================================================
# 第一部分：定义工具（Tools）
# ============================================================
# 改造要点：
#   1. get_weather → 接入真实天气 API（wttr.in，免费无需注册）
#   2. calculate  → 保留数学计算功能
#   3. 新增 get_exchange_rate → 实时汇率查询

@tool
def get_weather(city: str) -> str:
    """查询某个城市的实时天气。city 参数是城市名称（中文或英文均可）。
    示例：get_weather('Beijing') 或 get_weather('上海')"""
    try:
        # wttr.in 是一个免费天气 API，无需 API Key，支持全球城市
        resp = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},  # j1 = JSON 格式
            timeout=10,
        )
        data = resp.json()
        current = data["current_condition"][0]
        return (
            f"{city}: {current['weatherDesc'][0]['value']}，"
            f"温度 {current['temp_C']}°C，体感 {current['FeelsLikeC']}°C，"
            f"湿度 {current['humidity']}%，风速 {current['windspeedKmph']} km/h"
        )
    except Exception as e:
        return f"获取 {city} 天气失败：{e}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。例如：'2 + 3 * 4' 或 '100 / 5'。"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def get_exchange_rate(from_currency: str, to_currency: str = "CNY", amount: float = 1.0) -> str:
    """查询实时汇率。
    参数：
        from_currency: 源货币代码，如 USD、EUR、JPY、GBP
        to_currency:   目标货币代码，默认 CNY（人民币）
        amount:        要兑换的金额，默认 1
    示例：get_exchange_rate('USD', 'CNY', 100) 查询 100 美元换多少人民币"""
    try:
        # open.er-api.com：免费、无需注册、支持 160+ 货币
        resp = requests.get(
            f"https://open.er-api.com/v6/latest/{from_currency.upper()}",
            timeout=10,
        )
        data = resp.json()
        rate = data["rates"][to_currency.upper()]
        converted = round(amount * rate, 2)
        return (
            f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()}"
            f"（实时汇率，1 {from_currency.upper()} = {rate} {to_currency.upper()}）"
        )
    except Exception as e:
        return f"查询汇率失败：{e}"


# ============================================================
# 第二部分：初始化 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url="https://api.deepseek.com",
)


# ============================================================
# 第三部分：创建 Agent
# ============================================================
# debug=True:  开启 LangGraph 底层节点执行日志
# middleware:  注入自定义日志中间件，中文展示推理过程

logging_mw = LoggingMiddleware()

agent = create_agent(
    model=llm,
    tools=[get_weather, calculate, get_exchange_rate],
    system_prompt="你是一个智能助手，可以：1) 查询全球城市实时天气 2) 进行数学计算 3) 查询实时汇率。用中文回答，简洁友好。",
    middleware=[logging_mw],
    debug=True,
)


# ============================================================
# 第四部分：运行 Agent
# ============================================================

def ask(question: str):
    """提问并展示结果"""
    print(f"\n{'='*50}")
    print(f"🙋 问题：{question}")
    print('='*50)
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    final_msg = result["messages"][-1]
    print(f"\n✅ 最终回答：{final_msg.content}")


if __name__ == "__main__":
    print("🚀 LangChain 进阶 Agent 启动")
    print("工具列表：实时天气查询 | 数学计算 | 实时汇率查询\n")

    # 测试 1：天气（真实 API）
    ask("北京今天天气怎么样？")

    # 测试 2：汇率（新工具）
    ask("100 美元等于多少人民币？")

    # 测试 3：组合使用 —— 需要 Agent 自己决定用哪个工具
    ask("上海和东京，哪个城市现在温度更高？")

    # 测试 4：汇率 + 计算组合
    ask("如果我有 500 欧元，换成人民币是多少？然后除以 7 是多少？")
