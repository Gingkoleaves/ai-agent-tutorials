# ============================================================
# 第 4 篇：AutoGen —— 两个 AI 对话完成代码任务
# 运行方式：python 01_code_review.py
# ============================================================

# ---------- Python 基础说明 ----------
# 字典（dict）嵌套：字典里面再放字典
# config = {
#     "model": "gpt-4o-mini",     <- 键: 值
#     "api_key": "sk-...",
# }
# 访问嵌套字典：config["model"] 得到 "gpt-4o-mini"
# -------------------------------------

import os
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# AutoGen 使用 LLM 配置字典
llm_config = {
    "model": "deepseek-chat",
    "api_key": os.environ.get("OPENAI_API_KEY"),   # 从环境变量读取 API Key
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
}


# ============================================================
# 第一部分：定义 Agent
# AssistantAgent：AI 助手（不能执行代码）
# UserProxyAgent：代理用户，可以执行代码并返回结果
# ============================================================

# 程序员 Agent：负责写代码
programmer = AssistantAgent(
    name="程序员",
    system_message="""你是一位 Python 专家。
你的任务是编写清晰、正确、有注释的 Python 代码。
每次只写一个完整的函数或脚本，代码块用 ```python ... ``` 包裹。
写完代码后说"请审查我的代码"。""",
    llm_config=llm_config,
)

# 代码审查员 Agent：负责审查代码
reviewer = AssistantAgent(
    name="审查员",
    system_message="""你是一位严格的代码审查员，专注于代码质量。
审查代码时，检查：
1. 逻辑是否正确
2. 边界情况是否处理
3. 代码是否清晰易读
4. 是否有潜在的 bug

如果代码没问题，回复"代码审查通过 ✅ TERMINATE"。
如果有问题，说明具体问题并要求修改。""",
    llm_config=llm_config,
)

# 文档撰写员 Agent：负责生成文档
document_writer = AssistantAgent(
    name="文档撰写员",
    system_message="""你是一位技术文档撰写专家。
当代码审查通过后，你需要：
1. 阅读最终版本的代码
2. 生成一个完整的 README 文档，包括：
   - 项目概述（一句话说明功能和用途）
   - 函数签名和参数说明（表格形式）
   - 行为说明（列举核心逻辑要点）
   - 输出示例（仿真实输出格式）
   - 使用示例（Python 代码片段）
   - 注意事项（列出3-5条）
   - 依赖说明
输出格式为 Markdown，完成后说"✅ TERMINATE"。""",
    llm_config=llm_config,
)

# UserProxyAgent：模拟用户，负责启动对话
# human_input_mode="NEVER"：不需要真人输入，全自动运行
# code_execution_config：是否允许执行代码（这里关闭，只做对话演示）
user_proxy = UserProxyAgent(
    name="用户",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,   # 用户不自动回复
    code_execution_config=False,    # 不执行代码（安全起见）
    default_auto_reply="",
)


# ============================================================
# 第二部分：两个 Agent 直接对话
# ============================================================

def two_agent_chat(task: str):
    """程序员和审查员的双人对话"""
    print(f"\n{'='*60}")
    print(f"📋 任务：{task}")
    print('='*60)

    # initiate_chat：启动对话
    # recipient：对话对象
    # message：第一条消息（任务描述）
    # max_turns：最多对话几轮
    user_proxy.initiate_chat(
        recipient=programmer,
        message=f"请完成以下编程任务：\n\n{task}",
        max_turns=4,    # 最多4轮对话（程序员写→审查员审→程序员改→审查员确认）
    )


# ============================================================
# 第三部分：三个 Agent 群组对话
# ============================================================

def group_chat_demo(task: str):
    """三个 Agent 的群组对话"""
    print(f"\n{'='*60}")
    print(f"👥 群组讨论任务：{task}")
    print('='*60)

    # 创建群组聊天
    group_chat = GroupChat(
        agents=[programmer, reviewer, document_writer, user_proxy],
        messages=[],          # 消息历史（初始为空）
        max_round=6,          # 最多6轮
        # speaker_selection_method：谁来发言的策略
        # "auto"：由 GroupChatManager 自动选择
        speaker_selection_method="auto",
    )

    # GroupChatManager：群组的主持人，决定谁该发言
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    user_proxy.initiate_chat(
        recipient=manager,
        message=f"团队任务：{task}\n请程序员先写代码，然后审查员审核。审查通过后，由文档撰写员生成 README。",
        max_turns=1,
    )


if __name__ == "__main__":
    print("🤖 AutoGen 多智能体协作演示\n")

    # 进阶演示：三Agent群组对话（程序员→审查员→文档撰写员）
    group_chat_demo(
        "写一个 Python 函数，比较不同复杂度排序算法的性能，并输出结果。"
        "要求：有类型注释、有文档字符串、处理 N<=0 的边界情况"
    )
