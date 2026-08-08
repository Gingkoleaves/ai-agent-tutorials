# ============================================================
# 第 3 篇：CrewAI —— 让多个 AI 角色像团队一样协作
# 运行方式：python 01_content_crew.py
# ============================================================

# ---------- Python 基础说明 ----------
# 类（Class）：一个"模板"，用来创建对象
#   class Dog:           <- 定义模板
#       def bark(self):  <- 定义行为（方法）
#           print("汪！")
#   my_dog = Dog()      <- 用模板创建一个对象（实例化）
#   my_dog.bark()       <- 调用对象的行为
#
# 这里的 Agent、Task、Crew 都是 CrewAI 定义好的类
# 我们只需要传入参数来"实例化"它们
# -------------------------------------

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 创建 LLM 实例
llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, base_url="https://api.deepseek.com")


# ============================================================
# 第一部分：定义 Agent（角色）
# 每个 Agent 有：角色、目标、背景故事
# 这些描述直接影响 AI 的行为风格
# ============================================================

researcher = Agent(
    role="资深科技研究员",
    goal="深入研究指定主题，收集准确、全面的信息和最新数据",
    backstory=(
        "你是一位有10年经验的科技研究员，擅长快速梳理复杂信息。"
        "你的报告以结构清晰、数据翔实著称，总能找到别人忽略的关键细节。"
    ),
    llm=llm,
    verbose=True,                # 显示思考过程
    allow_delegation=False,      # 是否允许把任务委派给其他 Agent
)

writer = Agent(
    role="科技内容写手",
    goal="将研究成果转化为通俗易懂、引人入胜的文章",
    backstory=(
        "你是一位科技博主，擅长把复杂的技术概念用生活化语言解释清楚。"
        "你的文章总能让技术小白也看得懂，同时保持内容的准确性。"
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

editor = Agent(
    role="主编",
    goal="审核文章质量，确保内容准确、结构清晰、表达流畅",
    backstory=(
        "你是一位严格的主编，有20年媒体经验。"
        "你总能发现文章中的逻辑漏洞、表达不清和事实错误，并给出明确的修改意见。"
    ),
    llm=llm,
    verbose=True,
    allow_delegation=True,       # 主编可以把工作委派回去
)


# ============================================================
# 第二部分：定义 Task（任务）
# 每个任务绑定到一个 Agent，并描述期望的输出
# ============================================================

def create_tasks(topic: str):
    """根据主题创建任务列表"""

    research_task = Task(
        description=f"""
请研究以下主题：「{topic}」

你需要收集：
1. 核心定义和工作原理（3-5个关键点）
2. 实际应用场景（至少2个真实案例）
3. 当前局限性或挑战（2-3个）
4. 最新发展趋势（2025-2026年）

输出格式：结构化的研究笔记，分4个章节，总计400字左右。
""",
        expected_output="包含4个章节的结构化研究笔记，每章有标题和详细内容",
        agent=researcher,        # 这个任务由 researcher 执行
    )

    write_task = Task(
        description=f"""
基于研究员提供的资料，写一篇关于「{topic}」的科普文章。

要求：
- 开头用一个有趣的场景或问题引入（不超过3句话）
- 核心内容：3个小节，每节150字左右
- 结尾：一句话的行动建议
- 风格：通俗易懂，可以用类比，避免术语堆砌
- 总字数：500-600字
""",
        expected_output="500-600字的科普文章，包含引言、3个主体段落和结语",
        agent=writer,            # 由 writer 执行
        context=[research_task], # context：依赖哪些任务的输出作为输入
    )

    edit_task = Task(
        description=f"""
审核以下关于「{topic}」的文章，从以下维度评估：

1. 准确性：内容是否有明显错误？
2. 可读性：对非技术读者是否友好？
3. 结构性：逻辑是否清晰？
4. 吸引力：开头是否有钩子？结尾是否有力？

如果文章评分低于8分（满分10分），请直接给出修改后的版本。
如果高于8分，输出"✅ 文章质量优秀"并附上简短点评。
""",
        expected_output="审核报告：评分、各维度评价，以及修改版本（如需要）",
        agent=editor,
        context=[write_task],
    )

    return [research_task, write_task, edit_task]


# ============================================================
# 第三部分：组建 Crew（团队）并运行
# ============================================================

def run_crew(topic: str):
    print(f"\n{'='*60}")
    print(f"🚀 启动内容创作团队")
    print(f"📌 主题：{topic}")
    print('='*60)

    tasks = create_tasks(topic)

    crew = Crew(
        agents=[researcher, writer, editor],  # 团队成员
        tasks=tasks,                           # 任务列表
        process=Process.sequential,            # sequential：按顺序执行
        verbose=True,                          # 显示执行过程
    )

    # crew.kickoff()：启动团队，开始执行
    result = crew.kickoff()

    print(f"\n{'='*60}")
    print("🎉 最终输出：")
    print('='*60)
    print(result)


if __name__ == "__main__":
    run_crew("AI Agent 的记忆机制")
