# ============================================================
# CrewAI 进阶：Hierarchical 模式 + 真实联网搜索
# 运行方式：export SERPER_API_KEY="你的Key" && python 02_advanced_crew.py
#
# 与 01 的关键区别：
#   1. Process.hierarchical → Manager Agent 动态分配任务
#   2. backstory 更具体 → 强调数据来源、引用规范
#   3. expected_output 更严格 → 要求标注来源、URL、数据出处
#   4. SerperDevTool → 研究员可联网搜索，数据来自真实互联网
# ============================================================

import os
import sys

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# ============================================================
# API Key 检查
# ============================================================

# 需要两个 Key：DeepSeek（LLM）+ Serper（搜索）
if not os.getenv("SERPER_API_KEY"):
    print("⚠️  请设置 SERPER_API_KEY 环境变量：")
    print("   1. 访问 https://serper.dev 免费注册（2500次/月）")
    print('   2. export SERPER_API_KEY="你的Key"')
    print("   3. 重新运行此脚本")
    sys.exit(1)

# CrewAI 1.15+ 使用内置 LLM 类
llm = LLM(
    model="openai/deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# 初始化联网搜索工具
search_tool = SerperDevTool()


# ============================================================
# 第一部分：定义 Agent（角色）
#
# 🔄 修改点 vs 01：
#   - backstory 更具体，强调"必须使用搜索工具""引用真实来源"
#   - allow_delegation=True（hierarchical 模式要求）
#   - 研究员增加了 tools=[search_tool]
# ============================================================

researcher = Agent(
    role="资深科技研究员（联网版）",
    goal="使用联网搜索工具，从互联网获取关于指定主题的真实、最新信息，绝不编造数据",
    backstory=(
        "你是一位有10年经验的科技研究员，你最大的特点是从不依赖记忆——"
        "你总是先用搜索工具查找真实资料，核实每一个数据点。"
        "你的研究报告以「每条结论都有出处」著称，任何读者都能追溯到原始来源。"
        "在开始研究前，你会先搜索主题的最新进展（2025-2026年），确保信息不过时。"
    ),
    tools=[search_tool],         # 🔑 关键：接入真实联网搜索
    llm=llm,
    verbose=True,
    allow_delegation=True,       # hierarchical 模式允许委派
)

writer = Agent(
    role="数据驱动型科技作者",
    goal="基于研究员提供的真实资料，撰写有数据支撑、有来源可查的高质量科普文章",
    backstory=(
        "你是一位擅长用数据讲故事的科技作者。你不满足于泛泛而谈——"
        "每段论述都要求有具体数字、案例或研究来源作为支撑。"
        "你信奉「没有来源的断言只是观点」，因此你会在文章中自然地嵌入数据出处，"
        "让读者感觉像是在读一篇经过严谨调查的深度报道，而非 AI 臆造的文本。"
        "你擅长把研究报告中的数据和案例，转化为生动的叙述。"
    ),
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

editor = Agent(
    role="严格的事实核查主编",
    goal="逐条核查文章中的事实声明是否与原始研究数据一致，确保零AI幻觉",
    backstory=(
        "你是一位以铁面无私著称的主编，曾在顶级科技媒体担任事实核查主管15年。"
        "你对「AI 幻觉」零容忍——任何没有来源支撑的断言都必须被标记。"
        "你的审核清单包括：① 每个数据点是否可追溯到研究笔记中的来源？"
        "② URL/来源是否真实而非编造？③ 逻辑链是否完整？"
        "你宁可让文章重写，也不让未经验证的内容发布。"
    ),
    llm=llm,
    verbose=True,
    allow_delegation=True,
)


# ============================================================
# 第二部分：定义 Task（任务）
#
# 🔄 修改点 vs 01：
#   - expected_output 增加了"来源标注""URL""数据引用"等硬性要求
#   - description 更详细，给出了具体的输出格式规范
# ============================================================

def create_tasks(topic: str):
    research_task = Task(
        description=f"""
请使用联网搜索工具，研究以下主题：「{topic}」

你必须通过搜索工具（而非记忆）获取信息。请按以下框架组织：

1. **核心定义与原理**（搜索："{topic} 定义 工作原理"）
   - 3-5个关键概念，每个注明来源

2. **真实应用案例**（搜索："{topic} 应用案例 2025"）
   - 至少3个有具体公司/产品名称的真实案例
   - 每个案例注明来源URL或出处

3. **局限性/挑战**（搜索："{topic} 局限性 挑战"）
   - 2-3个已被报道或研究的问题

4. **最新趋势**（搜索："{topic} 2025 2026 趋势"）
   - 最近12个月的重要进展，注明时间

⚠️ 重要：如果搜索不到某个信息，请如实标注"未找到"，不要编造。
""",
        expected_output=(
            "结构化研究笔记（≥500字），包含4个章节，"
            "每条关键信息必须标注来源（格式：【来源：URL/出版物名称】），"
            "案例必须包含具体公司名称和数据，不可使用模糊描述"
        ),
        agent=researcher,
    )

    write_task = Task(
        description=f"""
基于研究员提供的研究笔记，写一篇数据驱动的科普文章。

要求（与普通文章的区别）：
- 开头：用一个最新的具体数据或事件引入（来自研究笔记，注明来源）
- 主体3小节，每节约200字：
  * 每小节至少包含1个具体数据点和1个真实案例
  * 自然地嵌入来源信息（如"据XX公司2025年发布的数据..."）
- 结尾：基于数据的一个趋势判断或行动建议
- 总字数：600-800字
- 风格：像《连线》或《经济学人》的科技报道，严谨但有阅读快感
""",
        expected_output=(
            "600-800字的深度科普文章，"
            "文中至少包含5处数据引用（标注来源），"
            "不少于3个带公司/机构名称的真实案例，"
            "无模糊表述（如'很多人认为''有研究表明'等）"
        ),
        agent=writer,
        context=[research_task],
    )

    edit_task = Task(
        description=f"""
对文章进行严格的事实核查和编辑审核。

核查清单（逐项检查）：
1. **事实准确性**：文中每个数据点是否能追溯到研究笔记中的具体来源？逐一比对。
2. **来源真实性**：所有URL、公司名、出版物名称是否为真实存在（非AI编造）？
3. **逻辑完整性**：从数据到结论的推理链是否完整？
4. **表达品质**：是否存在"很多人认为""有研究表明""众所周知"等无来源表述？

审核结论格式：
- 总分（/10）
- 各项评分（/10）
- 被标记的问题（逐条列出，指明段落和数据）
- 修改建议或修改后的全文
""",
        expected_output=(
            "审核报告（≥300字）：包含总分、分项评分、逐条事实核查结果、"
            "至少指出3处需要修改的具体问题（如有），"
            "若文章<8分别给出完整修改版"
        ),
        agent=editor,
        context=[write_task],
    )

    return [research_task, write_task, edit_task]


# ============================================================
# 第三部分：组建 Hierarchical Crew
#
# 🔄 关键区别 vs 01：
#   Process.hierarchical → CrewAI 自动创建 Manager Agent
#   Manager 根据 agent 的 role/backstory 动态分配任务
#   manager_llm 指定 Manager 使用的模型
# ============================================================

def run_crew(topic: str):
    print(f"\n{'='*60}")
    print(f"🚀 启动 Hierarchical 内容创作团队")
    print(f"📌 主题：{topic}")
    print(f"🔍 模式：Hierarchical（Manager 动态调度）")
    print(f"🌐 搜索：SerperDevTool（联网搜索已启用）")
    print('='*60)

    tasks = create_tasks(topic)

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=tasks,
        process=Process.hierarchical,       # 🔑 Manager Agent 自动调度
        manager_llm=llm,                    # Manager 用的模型
        verbose=True,
    )

    result = crew.kickoff()

    print(f"\n{'='*60}")
    print("🎉 最终输出：")
    print('='*60)
    print(result)


if __name__ == "__main__":
    run_crew("AI Agent 的记忆机制")
