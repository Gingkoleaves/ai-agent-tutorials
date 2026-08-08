# ============================================================
# Agno 进阶：角色扮演 + 自定义工具
# 运行方式：python 02_custom_tools.py
# ============================================================

import datetime
import json
import os
from pathlib import Path

import requests
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool


# ============================================================
# 第一部分：自定义工具（用 @tool 装饰器包装 Python 函数）
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询城市实时天气。
    city: 城市名称（中文或英文，如 'Beijing'、'上海'）
    数据来源：wttr.in 免费天气 API"""
    try:
        resp = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            timeout=10,
        )
        data = resp.json()
        c = data["current_condition"][0]
        return (
            f"🌤️ {city}: {c['weatherDesc'][0]['value']}，"
            f"温度 {c['temp_C']}°C，体感 {c['FeelsLikeC']}°C，"
            f"湿度 {c['humidity']}%，风速 {c['windspeedKmph']} km/h"
        )
    except Exception as e:
        return f"获取天气失败：{e}"


@tool
def read_local_file(filepath: str) -> str:
    """读取本地文件内容（文本文件）。
    filepath: 文件的绝对路径或相对路径
    注意：仅支持文本文件，二进制文件会报错"""
    try:
        p = Path(filepath).expanduser().resolve()
        if not p.exists():
            return f"文件不存在：{p}"
        if p.stat().st_size > 1024 * 100:  # 100KB 限制
            return f"文件过大（{p.stat().st_size} 字节），仅返回前 1000 字符：\n{p.read_text(encoding='utf-8')[:1000]}..."
        content = p.read_text(encoding="utf-8")
        return f"📄 {p.name}（{p.stat().st_size} 字节）：\n{content}"
    except UnicodeDecodeError:
        return f"无法读取文件 {filepath}：不是文本文件或编码非 UTF-8"
    except Exception as e:
        return f"读取文件失败：{e}"


@tool
def http_request(url: str, method: str = "GET") -> str:
    """发送 HTTP 请求（模拟调用公司内部 API）。
    url: 请求地址
    method: 请求方法（GET/POST，默认 GET）"""
    try:
        method = method.upper()
        if method not in ("GET", "POST"):
            return f"不支持的请求方法：{method}（仅支持 GET/POST）"

        resp = requests.request(method, url, timeout=10)
        resp.raise_for_status()

        # 尝试解析 JSON
        try:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return resp.text[:500]  # 非 JSON 则返回前500字符
    except requests.exceptions.Timeout:
        return f"请求超时：{url}（超过 10 秒无响应）"
    except requests.exceptions.ConnectionError:
        return f"连接失败：{url}（无法建立连接）"
    except Exception as e:
        return f"请求失败：{e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}（星期{['一','二','三','四','五','六','日'][now.weekday()]}）"


# ============================================================
# 第二部分：角色演示 —— 同一套工具、不同的 instructions
# ============================================================

# 公用工具包
common_tools = [get_weather, read_local_file, http_request]
# 注意：DeepSeek 不支持 `developer` 角色，必须显式设置 role_map
# 否则 agno 默认将 system → developer，导致 400 错误
model = OpenAIChat(
    id="deepseek-chat",
    base_url="https://api.deepseek.com",
    role_map={
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    },
)


def create_customer_service_agent() -> Agent:
    """客服助手：专业、有同理心、解决问题导向"""
    return Agent(
        model=model,
        tools=common_tools,
        instructions=[
            "你的角色：电商平台的高级客服专员，负责处理客户的各类问题",
            "核心原则：",
            "- 始终用「您」称呼客户，语气热情但不过分谄媚",
            "- 先共情再解决：先表达理解，再给出方案",
            "- 如果客户的问题涉及物流/退换货，主动提供操作步骤",
            "- 每次答复结尾，问一个确认性问题（如'这样能解决您的问题吗？'）",
            "你可以使用以下工具辅助服务：",
            "- get_weather: 查询客户所在地天气（如客户提到发货延迟，可解释天气影响）",
            "- read_local_file: 读取客户上传的文件",
            "- http_request: 查询内部系统数据（如订单状态、库存）",
            "用中文回复，每条回复 3-5 句话，简洁专业。",
        ],
        markdown=True,
    )


def create_translator_agent() -> Agent:
    """翻译官：精准翻译 + 语言文化注解"""
    return Agent(
        model=model,
        tools=common_tools,
        instructions=[
            "你的角色：精通中、英、日三语的专业翻译官，有10年跨文化沟通经验",
            "工作规范：",
            "- 当用户输入非中文内容时，自动检测语言并翻译成中文",
            "- 当用户输入中文并要求翻译时，翻译成用户指定的目标语言",
            "- 翻译原则：信（准确）> 达（通顺）> 雅（优美）",
            "- 关键术语或可能产生歧义的地方，用括号标注原文",
            "- 如果涉及文化特定概念（如'春节''寿司''Black Friday'），加简短的译者注",
            "- 可以调用 http_request 查询在线词典或术语库辅助翻译",
            "格式：",
            "【原文】（检测到的语言：XX）",
            "原文内容",
            "【译文】",
            "翻译内容",
            "【译者注】（如有必要）",
            "用中文回复（翻译结果除外）。",
        ],
        markdown=True,
    )


def create_data_analyst_agent() -> Agent:
    """数据分析师：用数据说话，结构化解构问题"""
    return Agent(
        model=model,
        tools=common_tools + [get_current_time],
        instructions=[
            "你的角色：资深数据分析师，擅长从原始数据中提取洞察",
            "工作方式：",
            "- 面对任何问题，先拆解成可量化的子问题",
            "- 优先使用数据说话，避免主观判断",
            "- 结论以结构化格式呈现：问题 → 数据 → 分析 → 建议",
            "- 如果需要外部数据验证，使用 http_request 请求相关 API",
            "- 如果用户提供了文件路径，使用 read_local_file 读取并分析内容",
            "输出格式偏好：",
            "- 使用 Markdown 表格呈现对比数据",
            "- 关键数字用 **加粗** 突出",
            "- 每个分析段落以一句话的结论开头（金字塔原理）",
            "用中文回复，冷静、客观但不冷漠。",
        ],
        markdown=True,
    )


# ============================================================
# 第三部分：运行演示
# ============================================================

def demo_role_switching():
    """演示：同一套工具 + 不同 instructions = 完全不同的行为"""
    print("\n" + "=" * 60)
    print("🎭 角色演示：instructions 如何塑造 Agent 的行为")
    print("=" * 60)

    # --- 场景1：客服助手 ---
    print("\n" + "─" * 40)
    print("👔 角色1：客服助手")
    print("─" * 40)

    cs_agent = create_customer_service_agent()
    cs_agent.print_response(
        "我买的手机壳已经5天了还没到货，上海这几天是不是有什么情况？帮我查查",
        stream=False,
    )

    # --- 场景2：翻译官 ---
    print("\n" + "─" * 40)
    print("🌐 角色2：翻译官")
    print("─" * 40)

    translator = create_translator_agent()
    translator.print_response(
        "请把下面这段话翻译成英文和日文：\n"
        "'人工智能正在深刻改变我们的工作方式，但人类的创造力和同理心仍然是不可替代的。'",
        stream=False,
    )

    # --- 场景3：数据分析师 ---
    print("\n" + "─" * 40)
    print("📊 角色3：数据分析师")
    print("─" * 40)

    analyst = create_data_analyst_agent()
    analyst.print_response(
        "我们公司上季度三个产品的销量分别是：A产品 1200件（环比+15%），"
        "B产品 850件（环比-8%），C产品 2100件（环比+3%）。"
        "请分析这三个产品的表现，给出策略建议。",
        stream=False,
    )


def demo_custom_tools():
    """演示：自定义工具的实际调用"""
    print("\n" + "=" * 60)
    print("🔧 工具演示：自定义 @tool 的实际效果")
    print("=" * 60)

    agent = Agent(
        model=model,
        tools=[get_weather, read_local_file, http_request],
        instructions=[
            "你是一个技术助手，擅长使用各种工具完成任务。",
            "当用户请求查询天气、读取文件或调用 API 时，请使用对应的工具。",
            "如果工具返回错误，友好地告诉用户并建议解决方案。",
            "用中文回复。",
        ],
        markdown=True,
    )

    # 测试1：天气工具
    print("\n─" * 40)
    print("🔧 测试1：get_weather")
    print("─" * 40)
    agent.print_response("北京现在天气怎么样？", stream=False)

    # 测试2：文件读取工具
    print("\n─" * 40)
    print("🔧 测试2：read_local_file")
    print("─" * 40)
    # 读取自身代码作为演示
    agent.print_response(
        "请用 read_local_file 工具读取 /media/gingkoleaves/Data/Tongji/Cheng/5/ai-agent-tutorials/05-agno/requirements.txt 文件，然后总结里面说了什么",
        stream=False,
    )

    # 测试3：HTTP API 工具
    print("\n─" * 40)
    print("🔧 测试3：http_request")
    print("─" * 40)
    agent.print_response(
        "帮我用 http_request 访问 https://api.github.com/repos/agno-agi/agno 这个地址，"
        "告诉我这个项目有多少 star 和 fork",
        stream=False,
    )


if __name__ == "__main__":
    demo_role_switching()
    demo_custom_tools()
