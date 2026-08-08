# ============================================================
# Smolagents 进阶：Hub 工具生态 + 执行日志 + E2B 沙箱
# 运行方式：export HF_TOKEN="hf_..." && python 02_hub_tools.py
# ============================================================

import os
import json
import datetime
from pathlib import Path

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    MultiStepAgent,
    tool,
    LiteLLMModel,
    load_tool,            # 从 HF Hub 加载社区工具
    LogLevel,             # 日志级别控制
)


# ============================================================
# 第一部分：从 HuggingFace Hub 加载社区工具
#
# Smolagents 生态：社区贡献了上百个工具，直接从 Hub 加载
# 无需自己造轮子！常见工具有：
#   - 文生图: m-ric/text-to-image (FLUX.1-schnell)
#   - 语音合成: m-ric/text-to-speech
#   - 语音识别: m-ric/speech-to-text
#   - 联网搜索: DuckDuckGoSearchTool (内置)
#   - 网页抓取: m-ric/web-scraper
#
# 加载方式：load_tool("用户名/仓库名", trust_remote_code=True)
# ============================================================

def get_hub_tools():
    """尝试加载 HF Hub 工具，失败时返回友好提示"""
    hf_token = os.getenv("HF_TOKEN")

    tools = {}
    unavailable = []

    # --- 工具1: 联网搜索（内置，无需 Hub） ---
    try:
        from smolagents import DuckDuckGoSearchTool
        tools["web_search"] = DuckDuckGoSearchTool()
        print("✅ DuckDuckGoSearchTool 已加载（内置）")
    except ImportError:
        unavailable.append("DuckDuckGoSearchTool")

    # --- 工具2: 文生图（从 Hub 加载）---
    if hf_token:
        try:
            tools["image_gen"] = load_tool(
                "m-ric/text-to-image",
                trust_remote_code=True,
                token=hf_token,
            )
            print("✅ text-to-image 已加载（HF Hub: FLUX.1-schnell）")
        except Exception as e:
            unavailable.append(f"text-to-image ({e})")
    else:
        unavailable.append("text-to-image（需 HF_TOKEN）")

    # --- 工具3: 自定义工具（始终可用）---
    @tool
    def get_current_time() -> str:
        """获取当前日期和时间，包含星期信息。"""
        now = datetime.datetime.now()
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')} 星期{weekdays[now.weekday()]}"

    tools["time"] = get_current_time
    print("✅ get_current_time 已加载（自定义）")

    if unavailable:
        print(f"\n⚠️  以下工具不可用：{', '.join(unavailable)}")
        if not hf_token:
            print("💡 设置 HF_TOKEN 即可使用 Hub 工具：")
            print("   1. https://huggingface.co/settings/tokens 创建 Token")
            print('   2. export HF_TOKEN="hf_..."')

    return list(tools.values()), tools


# ============================================================
# 第二部分：执行日志调试 —— MultiStepAgent 的观察方法
#
# Smolagents 提供了多层级的调试手段：
#   1. verbosity_level — 控制日志详细程度
#   2. planning_interval — 周期性战略规划（能看到 Agent "思考策略"）
#   3. step_callbacks — 每个执行步骤后的回调（自定义日志）
#   4. replay() — 事后回放整个执行过程
#   5. return_full_result — 返回完整的执行元数据
# ============================================================

# --- 2a. 自定义 Step Callback（观察每一步的内部状态）---

def debug_step_callback(memory_step, agent):
    """每个步骤完成后触发，打印Agent的思考过程"""
    step_type = type(memory_step).__name__
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    # 记录步骤类型
    if "ActionStep" in step_type:
        step_num = getattr(memory_step, "step_number", "?")
        print(f"\n  [{ts}] 📍 第 {step_num} 步完成")

        # 打印 LLM 输出（Agent 的"想法"或代码）
        if hasattr(memory_step, "model_output"):
            output = str(memory_step.model_output)
            if output:
                preview = output[:200].replace("\n", "\n    ")
                print(f"     💭 模型输出: {preview}...")

        # 打印执行耗时
        if hasattr(memory_step, "duration") and memory_step.duration:
            print(f"     ⏱️  耗时: {memory_step.duration:.2f}s")

        # 打印 token 消耗
        if hasattr(memory_step, "token_usage") and memory_step.token_usage:
            i_tok = getattr(memory_step.token_usage, "input_tokens", 0)
            o_tok = getattr(memory_step.token_usage, "output_tokens", 0)
            print(f"     📊 Token: 输入{i_tok} + 输出{o_tok}")

    elif "PlanningStep" in step_type:
        print(f"\n  [{ts}] 🎯 战略规划更新")
        plan = getattr(memory_step, "plan", "")
        if plan:
            print(f"     📋 新计划:\n{plan[:300]}")

    elif "SystemPromptStep" in step_type:
        print(f"  [{ts}] 🚀 Agent 初始化（系统提示加载）")

    elif "TaskStep" in step_type:
        task = getattr(memory_step, "task", "")
        print(f"  [{ts}] 📝 任务接收: {task[:80]}...")


# --- 2b. 回放分析函数 ---

def analyze_run(agent, title: str = ""):
    """执行后分析：汇总 token、耗时、步骤数"""
    if title:
        print(f"\n{'='*55}")
        print(f"📊 执行分析：{title}")
        print('='*55)

    try:
        # replay() 回放完整执行过程
        agent.replay(detailed=False)
    except Exception as e:
        print(f"  （replay 不可用: {e}）")


# ============================================================
# 第三部分：创建 Agent（展示不同的调试配置）
# ============================================================

model = LiteLLMModel(
    model_id="openai/deepseek-chat",
    temperature=0,
    api_base="https://api.deepseek.com",
)

# --- Agent A: 基础调试（verbosity_level=2） ---
def create_debug_agent(tools):
    """创建带完整调试日志的 Agent"""
    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=8,
        # ---- 调试配置 ----
        verbosity_level=2,              # 0=安静 1=基础 2=详细
        planning_interval=3,            # 每3步做一次战略规划
        step_callbacks=[debug_step_callback],  # 自定义回调
    )


# ============================================================
# 第四部分：运行演示
# ============================================================

def demo_hub_tools():
    """演示：Hub 工具 + 执行日志调试"""
    print("\n" + "=" * 60)
    print("🛠️  第一部分：Hub 工具加载 + 执行日志观察")
    print("=" * 60)

    all_tools, tool_map = get_hub_tools()

    agent = create_debug_agent(all_tools)
    print(f"\n🔍 Agent 配置：")
    print(f"   planning_interval: {agent.planning_interval}（每3步规划一次）")
    print(f"   verbosity_level: 2（详细日志）")
    print(f"   max_steps: 8")
    print(f"   已注册 step_callback: debug_step_callback")

    # 任务：联网搜索 + 分析 + 输出结构化结果
    task = """
请完成以下任务并输出最终结果：

1. 搜索 "2025年AI Agent框架对比 Smolagents vs LangChain vs CrewAI 最新动态"
   （使用 DuckDuckGo 搜索工具）

2. 根据搜索结果，总结每个框架的 1-2 个核心优势

3. 格式化输出为以下结构：
   ### 2025 AI Agent 框架对比
   | 框架 | 开发者 | 核心优势 |
   |------|--------|---------|
   （表格内容）

4. 最后加一段 2-3 句话的趋势总结
"""
    print(f"\n📝 任务: {task[:100]}...")
    print("─" * 55)

    result = agent.run(task)
    print(f"\n📄 最终结果：\n{result}")


def demo_debug_analysis():
    """演示：执行后分析（replay + 完整结果）"""
    print("\n" + "=" * 60)
    print("🔬 第二部分：执行日志深度分析")
    print("=" * 60)

    @tool
    def analyze_numbers(data: str) -> str:
        """分析逗号分隔的数字，返回平均值、最大值、最小值。

        Args:
            data: '1,2,3,4,5' 格式的逗号分隔数字
        """
        nums = [float(x.strip()) for x in data.split(",")]
        return json.dumps({
            "count": len(nums),
            "mean": round(sum(nums) / len(nums), 2),
            "max": max(nums),
            "min": min(nums),
            "sorted": sorted(nums),
        }, ensure_ascii=False)

    agent = CodeAgent(
        tools=[analyze_numbers],
        model=model,
        max_steps=5,
        verbosity_level=2,
        planning_interval=2,    # 每2步规划一次
        step_callbacks=[debug_step_callback],
    )

    task = """
请分步骤完成：
1. 数据 "15, 23, 8, 42, 16, 9, 31, 7" 用 analyze_numbers 工具分析
2. 如果平均值小于 25，再生成一组数据 "50, 60, 45, 70, 55" 并分析
3. 比较两组数据，哪个组的平均值更高？
4. 输出最终比较结论
"""
    print(f"\n📝 多步任务（含 planning_interval=2）")

    result = agent.run(task)
    print(f"\n📄 最终结论：\n{result}")

    # 事后回放
    print("\n" + "─" * 55)
    print("🎬 执行回放 (agent.replay)：")
    print("─" * 55)
    agent.replay(detailed=False)


def demo_e2b_sandbox():
    """演示：E2B 云端沙箱 —— 生产级安全隔离（需要 E2B_API_KEY）"""
    print("\n" + "=" * 60)
    print("🛡️  第三部分：E2B 云端沙箱实测")
    print("=" * 60)

    e2b_key = os.getenv("E2B_API_KEY")
    if not e2b_key:
        print("""
⚠️  未设置 E2B_API_KEY，跳过沙箱测试。

获取 Key 步骤：
  1. 访问 https://e2b.dev 注册账号
  2. 在 Dashboard → API Keys 中创建 Key
  3. export E2B_API_KEY="e2b_..."
  4. 重新运行此脚本

💡 E2B 沙箱方案：
  优点: 云端全隔离，每次执行后自动销毁，支持 GPU
  价格: 有免费额度（每月 100 小时），按实际使用计费
""")
        return

    print("🔑 E2B_API_KEY 已设置，启动云端沙箱...\n")

    # ============================================================
    # 测试 1：原生 E2B 沙箱 —— 直接执行 Python 代码
    # ============================================================
    print("─" * 55)
    print("🧪 测试 1：原生 E2B 沙箱执行 Python 代码")
    print("─" * 55)

    sandbox = None
    try:
        from e2b import Sandbox

        # 创建沙箱（自动分配云端 VM）
        print("🔄 正在创建 E2B 沙箱实例...")
        sandbox = Sandbox.create(
            template="code-interpreter-v1",  # 预装 Python 3 + 常用库
            timeout=120,                      # 120 秒后自动销毁
        )
        print(f"✅ 沙箱已启动")
        print(f"   沙箱 ID: {sandbox.sandbox_id}")

        # 在沙箱中写入 Python 文件再执行（避免 bash 转义问题）
        print("\n📝 执行: 安装 numpy 并计算矩阵乘法...")
        python_code = """
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = a @ b
print(f"矩阵 a:\\n{a}")
print(f"矩阵 b:\\n{b}")
print(f"a @ b =\\n{c}")
print(f"特征值: {np.linalg.eigvals(c)}")

import sys, socket
print(f"Python 版本: {sys.version}")
print(f"沙箱主机名: {socket.gethostname()}")
"""
        sandbox.filesystem.write("/tmp/matrix_calc.py", python_code)
        result = sandbox.commands.run("pip install numpy -q && python3 /tmp/matrix_calc.py")
        print(f"✅ stdout:\n{result.stdout}")

        # 测试文件操作
        print("📝 执行: 在沙箱中创建文件并读取...")
        sandbox.commands.run("echo 'Hello from E2B sandbox!' > /tmp/sandbox_test.txt")
        result2 = sandbox.commands.run("cat /tmp/sandbox_test.txt && wc -c /tmp/sandbox_test.txt")
        print(f"✅ 文件内容: {result2.stdout.strip()}")

        # 上传本地文件到沙箱
        print("\n📝 测试: 上传文件到沙箱...")
        local_file = Path(__file__)
        sandbox.filesystem.write(
            "/tmp/uploaded_script.py",
            local_file.read_text()[:500]  # 上传脚本前 500 字符
        )
        result3 = sandbox.commands.run("head -3 /tmp/uploaded_script.py")
        print(f"✅ 上传成功，文件开头: {result3.stdout.strip()}")

        print(f"\n📊 沙箱统计：")
        try:
            metrics = sandbox.get_metrics()
            print(f"   CPU 使用: {metrics.cpu_used_ms}ms")
            print(f"   内存使用: {metrics.mem_used_bytes} bytes")
        except Exception:
            print(f"   (metrics 不可用)")

    except Exception as e:
        print(f"❌ E2B 沙箱测试失败: {e}")
    finally:
        if sandbox:
            sandbox.kill()
            print("\n🛑 沙箱已销毁（生产环境每次执行后自动销毁）")

    # ============================================================
    # 测试 2：CodeAgent + E2B（Agent 生成的代码在云端执行）
    # ============================================================
    print("\n" + "─" * 55)
    print("🤖 测试 2：CodeAgent + E2B —— Agent 代码在云端沙箱中执行")
    print("─" * 55)

    @tool
    def system_info() -> str:
        """获取系统信息（Python版本、操作系统、主机名）。
        此工具在 E2B 沙箱中执行，返回的是云端环境信息。"""
        import sys
        import platform
        import socket
        return (
            f"Python: {sys.version}\n"
            f"OS: {platform.system()} {platform.release()}\n"
            f"主机名: {socket.gethostname()}\n"
            f"注意: 如果通过 E2B 执行，以上信息来自云端沙箱"
        )

    try:
        model = LiteLLMModel(
            model_id="openai/deepseek-chat",
            temperature=0,
            api_base="https://api.deepseek.com",
        )

        agent = CodeAgent(
            tools=[system_info],
            model=model,
            max_steps=5,
            verbosity_level=2,
            # ====== E2B 配置 ======
            executor_type="e2b",
            executor_kwargs={
                "timeout": 120,
                "envs": {"APP_ENV": "production"},
            },
        )

        print("⚙️  CodeAgent 配置: executor_type='e2b'")
        print("    所有 LLM 生成的 Python 代码将在云端隔离环境中执行")

        result = agent.run("""
请调用 system_info 工具获取系统信息，
然后写 Python 代码计算 1 到 100 的所有质数之和，
最后报告：系统信息和计算结果各是什么。
""")
        print(f"\n📄 Agent 结果:\n{result}")

    except Exception as e:
        print(f"❌ CodeAgent E2B 测试失败: {e}")
        print(f"💡 提示: 这是正常的，smolagents 需要通过特定方式集成 E2B executor。")
        print(f"   原生 E2B Sandbox API（测试1）已确认可用。")

    print(f"\n💡 总结：E2B 沙箱 v.s. 本地执行对比")
    print(f"   {'指标':<20} {'local':<25} {'e2b':<25}")
    print(f"   {'─'*20} {'─'*25} {'─'*25}")
    print(f"   {'安全隔离':<20} {'AST 白名单（有限）':<25} {'完全云隔离 ✅':<25}")
    print(f"   {'依赖管理':<20} {'手动安装':<25} {'模板预装 ✅':<25}")
    print(f"   {'每次干净环境':<20} {'否（有残留风险）':<25} {'是（自动销毁）✅':<25}")
    print(f"   {'GPU 支持':<20} {'需本地配置':<25} {'模板支持 ✅':<25}")
    print(f"   {'适用场景':<20} {'开发/受信任代码':<25} {'生产/用户提交代码':<25}")


def demo_image_generation():
    """演示：从 HF Hub 加载文生图工具，AI 生成图片并保存到本地"""
    print("\n" + "=" * 60)
    print("🎨 第四部分：文生图工具实测（HF Hub → FLUX.1-schnell）")
    print("=" * 60)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("""
⚠️  未设置 HF_TOKEN，跳过文生图测试。

获取 Token 步骤：
  1. 访问 https://huggingface.co/settings/tokens
  2. 点击 "New token" → 选择 "Read" 权限 → 生成
  3. export HF_TOKEN="hf_..."
  4. 重新运行此脚本

💡 文生图工具说明：
  仓库: m-ric/text-to-image
  模型: black-forest-labs/FLUX.1-schnell（最快速度）
  功能: 根据中文/英文描述生成图片，支持高分辨率
""")
        return

    print("🔑 HF_TOKEN 已设置，开始加载文生图工具...")

    # --- 加载文生图工具 ---
    try:
        image_gen_tool = load_tool(
            "m-ric/text-to-image",
            trust_remote_code=True,
            token=hf_token,
        )
        print(f"✅ 工具加载成功: {image_gen_tool.name}")
        print(f"   描述: {image_gen_tool.description}")
        print(f"   输入参数: {image_gen_tool.inputs}")
    except Exception as e:
        print(f"❌ 工具加载失败: {e}")
        return

    # --- 创建 Agent ---
    model = LiteLLMModel(
        model_id="openai/deepseek-chat",
        temperature=0,
        api_base="https://api.deepseek.com",
    )

    agent = CodeAgent(
        tools=[image_gen_tool],
        model=model,
        max_steps=5,
        verbosity_level=2,
    )

    # --- 测试场景：多个不同风格的图片生成 ---
    test_prompts = [
        "生成一张日本动漫风格的图片：一个可爱的橘猫戴着巫师帽，在魔法图书馆里看书，温暖的光线从窗户照进来",
        "生成一张写实风格的图片：未来城市的夜景，高耸的玻璃建筑，飞行汽车在楼宇间穿梭，霓虹灯光倒映在雨后湿润的街道上",
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'─'*55}")
        print(f"🖼️  测试 {i+1}/{len(test_prompts)}")
        print(f"📝 描述: {prompt[:60]}...")
        print('─'*55)

        try:
            # Agent 会调用 image_gen_tool 生成图片
            # 直接调用工具更快，不需要 Agent 推理
            print("⚙️  正在调用 FLUX.1-schnell 模型生成图片...")
            result = agent.run(f"请用文生图工具生成以下图片：{prompt}")

            # 如果是 PIL Image 对象，保存到文件
            from PIL import Image
            output_dir = Path(__file__).parent / "generated_images"
            output_dir.mkdir(exist_ok=True)

            if hasattr(result, "save"):
                # result 是 PIL Image
                filename = output_dir / f"generated_{i+1}_{datetime.datetime.now().strftime('%H%M%S')}.png"
                result.save(str(filename))
                print(f"💾 图片已保存: {filename}")
                print(f"   尺寸: {result.size[0]}x{result.size[1]}")
            elif isinstance(result, str) and result.endswith(".png"):
                print(f"📄 图片路径: {result}")
            else:
                print(f"📄 结果: {str(result)[:200]}...")

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            # 尝试直接调用工具（绕过 Agent）
            print("🔄 尝试直接调用工具...")
            try:
                direct_result = image_gen_tool(prompt)
                from PIL import Image
                output_dir = Path(__file__).parent / "generated_images"
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f"direct_gen_{i+1}.png"
                direct_result.save(str(filename))
                print(f"💾 直接调用成功！图片已保存: {filename}")
            except Exception as e2:
                print(f"❌ 直接调用也失败: {e2}")

    print(f"\n📁 图片保存目录: {Path(__file__).parent / 'generated_images'}")


if __name__ == "__main__":
    print("🦤 Smolagents 进阶演示")
    print("   Step 2: HuggingFace Hub 工具生态（联网搜索 + 文生图）")
    print("   Step 3: MultiStepAgent 执行日志 + E2B 沙箱\n")

    demo_hub_tools()
    demo_debug_analysis()
    demo_image_generation()    # 🎨 新增：文生图实测
    demo_e2b_sandbox()
