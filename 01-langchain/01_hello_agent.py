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
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

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
# 第三部分：定义提示词模板（Prompt Template）
# Agent 需要一个"系统设定"告诉它该怎么工作
# ============================================================

# ChatPromptTemplate.from_messages：用消息列表定义对话结构
# ("system", "...") 是系统提示，告诉 AI 它的角色
# ("human", "{input}") 是用户输入，{input} 是占位符，会被实际内容替换
# ("placeholder", "{agent_scratchpad}") 是 Agent 的"草稿纸"，记录思考过程
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以查询天气和进行数学计算。用中文回答。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# ============================================================
# 第四部分：创建 Agent 和 AgentExecutor
# Agent = LLM + 工具列表 + 提示词模板（决策大脑）
# AgentExecutor = 运行 Agent 并执行工具的执行器（行动身体）
# ============================================================

tools = [get_weather, calculate]   # 列表：把工具放在方括号里

# create_tool_calling_agent：创建一个支持工具调用的 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor：让 Agent 真正跑起来
# verbose=True：打印详细的思考过程，方便学习理解
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ============================================================
# 第五部分：运行 Agent
# ============================================================

def ask(question: str):
    """提问并打印结果"""
    print(f"\n{'='*50}")
    print(f"🙋 问题：{question}")
    print('='*50)
    result = agent_executor.invoke({"input": question})
    print(f"\n✅ 最终回答：{result['output']}")


# if __name__ == "__main__"：
# 只有直接运行这个文件时才执行，import 时不会执行
# 这是 Python 的惯例写法
if __name__ == "__main__":
    ask("北京今天天气怎么样？")
    ask("上海和广州，哪个城市温度更高？")
    ask("如果我有 100 块钱，每天花 3.5 块，能花多少天？")
