# ============================================================
# 第 1 步：Hello Agent —— 最简单的 LangChain Agent
# 运行方式：python 01_hello_agent.py
# 需要：export OPENAI_API_KEY="sk-..."
# ============================================================

# ---------- Python 基础说明 ----------
# import：从某个包导入功能，就像"从工具箱里拿出一把锤子"
# from A import B：从 A 里面只拿 B，更精准
# -------------------------------------

from langchain_openai import ChatOpenAI          # OpenAI 的对话模型
from langchain_core.tools import tool            # 把函数变成 Agent 工具的装饰器
from langchain.agents import create_agent        # LangChain 1.3+ 新版 API

# ============================================================
# 第一部分：定义工具（Tools）
# @tool 是"装饰器"，作用是在函数上挂一个标签告诉 LangChain：
# "这个函数是 Agent 可以调用的工具"
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。city 参数是城市名称（中文或英文均可）。"""
    # 这里用字典模拟真实天气 API，实际项目可以接真实接口
    # 字典格式：{"键": "值"}，用键来查值
    weather_data = {
        "北京": "☀️ 晴天，25°C，微风",
        "上海": "🌤️ 多云，22°C，东南风",
        "广州": "🌧️ 小雨，28°C，湿度较高",
        "深圳": "⛅ 阴转晴，27°C",
        "成都": "🌫️ 有雾，18°C",
    }
    # dict.get(key, 默认值)：找到就返回值，找不到就返回默认值
    return weather_data.get(city, f"暂无 {city} 的天气数据，请检查城市名称")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。例如：'2 + 3 * 4' 或 '100 / 5'。"""
    # eval() 可以执行字符串里的数学表达式
    # 注意：生产环境中要限制 eval 的使用范围（安全考虑）
    try:
        result = eval(expression)           # 计算表达式
        return f"{expression} = {result}"   # f-string：把变量嵌入字符串的方式
    except Exception as e:
        return f"计算出错：{e}"             # Exception e：捕获错误信息


# ============================================================
# 第二部分：初始化大模型（LLM）
# ============================================================

# ChatOpenAI：连接 OpenAI 的对话模型
# model：用哪个模型，gpt-4o-mini 速度快且便宜
# temperature：创造力，0 表示"严格按事实说"，1 表示"天马行空"
llm = ChatOpenAI(model="deepseek-chat", temperature=0, base_url="https://api.deepseek.com")


# ============================================================
# 第三部分：系统提示词
# LangChain 1.3+ 直接传字符串，不需要 ChatPromptTemplate
# ============================================================
system_prompt = "你是一个智能助手，可以查询天气和进行数学计算。用中文回答。"


# ============================================================
# 第四部分：创建 Agent
# create_agent：LangChain 1.3+ 新版 API
# 传入 model + tools + system_prompt，返回一个可直接调用的 Agent（CompiledStateGraph）
# 不需要 AgentExecutor 了，Agent 本身就是可执行的
# ============================================================

tools = [get_weather, calculate]   # 列表：把工具放在方括号里

# create_agent：创建一个能调用工具的 Agent
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


# ============================================================
# 第五部分：运行 Agent
# ============================================================

# if __name__ == "__main__"：
# 只有直接运行这个文件时才执行，import 时不会执行
# 这是 Python 的惯例写法
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    def ask(question: str):
        """提问并打印结果"""
        print(f"\n{'='*50}")
        print(f"🙋 问题：{question}")
        print('='*50)
        # 新 API：传入 messages 列表，最后一条 AI 消息即为回答
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        # result["messages"] 是整个对话的消息列表，最后一条是 AI 的回答
        final_msg = result["messages"][-1]
        print(f"\n✅ 最终回答：{final_msg.content}")

    ask("北京今天天气怎么样？")
    ask("上海和广州，哪个城市温度更高？")
    ask("如果我有 100 块钱，每天花 3.5 块，能花多少天？")
