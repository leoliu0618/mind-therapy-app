import streamlit as st
import os
from openai import OpenAI
import json
import re
import pandas as pd # 导入 pandas
import random     # 导入 random

# 初始化 OpenAI 客户端
# 确保在运行前设置了 OPENAI_API_KEY 环境变量
# 例如: export OPENAI_API_KEY='你的api_key'
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 数据集加载函数 ---
@st.cache_data
def load_c2d2_data(filepath="C2D2_dataset.csv"):
    """从 CSV 文件加载 C2D2 数据集，尝试不同的编码"""
    encodings_to_try = ['gb18030', 'gbk', 'utf-8'] # 尝试的编码列表，优先尝试中文编码
    df = None
    error_messages = []

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc) # **在这里指定编码**
            st.success(f"成功使用 '{enc}' 编码加载 C2D2 数据集: {len(df)} 条记录")

            # **重要：确认这里的列名与您的文件一致**
            required_columns = ['场景', '标签'] # 确认这两个列名存在
            if not all(col in df.columns for col in required_columns):
                error_messages.append(f"使用 '{enc}' 编码加载成功，但缺少必要的列（需要 '{required_columns[0]}' 和 '{required_columns[1]}'）。")
                df = None # 标记为失败，尝试下一种编码
                continue # 继续尝试下一种编码

            # 可选：移除缺少关键信息的行
            df.dropna(subset=required_columns, inplace=True)
            if df.empty:
                 error_messages.append(f"使用 '{enc}' 编码加载并清理后数据为空。")
                 df = None # 标记为失败
                 continue # 继续尝试下一种编码

            # 如果成功加载且数据有效，则跳出循环
            return df

        except UnicodeDecodeError:
            error_messages.append(f"尝试使用 '{enc}' 编码失败 (UnicodeDecodeError)。")
            continue # 尝试列表中的下一个编码
        except FileNotFoundError:
            st.error(f"错误：无法找到数据集文件 '{filepath}'。请确保文件位于正确的位置。")
            return None # 文件不存在，直接返回 None
        except Exception as e:
            error_messages.append(f"加载数据集时发生其他错误 (编码 '{enc}'): {e}")
            # 对于其他未知错误，也可能需要停止尝试
            # return None # 标记为失败
    # 如果所有编码都尝试失败
    st.error("无法使用常见的编码 (gb18030, gbk, utf-8) 加载数据集。请检查文件内容或尝试指定其他编码。")
    st.error(f"尝试记录: {'; '.join(error_messages)}")
    return None # 所有尝试都失败，返回 None

# --- Prompt Templates (严格遵循论文描述和数据流) ---
PROMPT_TEMPLATES = {
    # Trigger (τ) - 接收 Pᵢ₋₁
    "trigger_0": """
你是一个情景再现师 (Trigger, τ)。
任务：根据主题 {theme} 和用户的初始担忧 {concerns} (W)，生成初始场景 (S₀)。
要求：
1. 场景应充分反映用户的状态和担忧。
2. 场景是故事背景，不含对话或心理描述。
3. 不含价值判断。
4. 输出格式：
Scene: <生成的初始场景 S₀，不超过150字>
""",
    "trigger_i": """
你是一个情景再现师 (Trigger, τ)。
任务：基于主题 {theme}，上一轮 (i-1) 用户的安慰 {comfort_prev} (Cᵢ₋₁)，以及上一轮策略师的规划 {progression_prev} (Pᵢ₋₁)，生成当前轮 (i) 的场景 (Sᵢ)。
上一轮策略师对本轮场景的指导 (来自 Pᵢ₋₁): {directive_scene}
要求：
1. 首先，请思考场景如何根据策略师的指导进行构建或调整，说明思考过程 (CoT)。
2. 然后，输出生成的场景 Sᵢ。
3. 场景要与历史发展和策略师指导一致。
4. 场景是故事背景，不含对话或心理描述。
5. 不含价值判断。
6. 输出格式：
思考过程：<你的思考>
Scene: <生成的场景 Sᵢ，不超过150字>
""",
    # Devil (δ) - 接收 Pᵢ₋₁
    "devil_0": """
你是一个模拟认知扭曲的患者 (Devil, δ)。
你的人格特质倾向: {personality_traits}
初始场景 (S₀): {scene}
你的初始担忧 (W): {concerns}
任务：基于场景和担忧，模拟第一人称视角，产生一个核心的初始负面想法 (D₀)，并说明其认知扭曲类型。
要求：
1. 想法要符合场景、担忧和人格特质。
2. 简短，像内心闪过的念头。
3. 输出格式：
Type: <认知扭曲类型>
Thoughts: <第一人称的初始想法 D₀，不超过30字>
""",
    # Devil (δ) - Round 0 - Using C2D2 Type (New)
    "devil_0_c2d2": """
你是一个模拟认知扭曲的患者 (Devil, δ)。
你的人格特质倾向: {personality_traits}
初始场景 (S₀): {scene}
你的初始担忧 (W): {concerns}
已知这个情境容易引发的认知扭曲类型是：{c2d2_distortion_type}
任务：基于场景、你的担忧、你的人格特质，并严格围绕指定的认知扭曲类型 "{c2d2_distortion_type}"，模拟第一人称视角，产生一个核心的初始负面想法 (D₀)。
要求：
1. 想法必须明确体现指定的认知扭曲类型 "{c2d2_distortion_type}"。
2. 想法要符合场景、担忧和人格特质。
3. 简短，像内心闪过的念头。
4. 输出格式（只需要想法，类型已知）：
Thoughts: <第一人称的初始想法 D₀，不超过30字>
""",
    "devil_i": """
你是一个模拟认知扭曲的患者 (Devil, δ)。
你的人格特质倾向: {personality_traits}
当前场景 (Sᵢ): {scene}
你的认知扭曲类型大致是: {type_prev}
上一轮 (i-1) 你的想法 (Dᵢ₋₁): {thought_prev}
上一轮 (i-1) 安慰者的话 (Cᵢ₋₁): {comfort_prev}
上一轮策略师对你本轮思想演变的指导 (来自 Pᵢ₋₁): {directive_thought}
任务：根据当前情境、人格特质、上一轮互动以及策略师的指导，模拟你此刻第一人称可能的想法 (Dᵢ)。这个想法应体现出策略师指导的思想演变方向（或固守）。
要求：
1. 想法要符合情境、人格、互动历史和指导方向。
2. 简短，像内心闪过的念头。
3. 输出格式：
Thoughts: <第一人称的想法 Dᵢ，不超过30字>
""",
    # Guide (g) - 输出 Gᵢ 和 Mᵢ
    "guide": """
你是一个专业的心理指导师 (Guide, g)。
当前场景 (Sᵢ): {scene}
患者当前的想法 (Dᵢ): {thoughts} (类型: {type})
任务：
1. 生成1-2条具体的、可操作的安慰引导建议 (Gᵢ)，帮助“安慰者”进行认知重构。
2. 基于当前场景 (Sᵢ) 和想法 (Dᵢ)，生成本回合的结构化记忆总结 (Mᵢ)。总结应包含场景关键点、想法核心、认知扭曲类型、潜在的情感基调。
要求：
1. 建议 (Gᵢ) 要紧密结合 Sᵢ 和 Dᵢ。
2. 记忆总结 (Mᵢ) 要简洁、结构化，捕捉本轮核心信息。
3. 输出必须是严格的 JSON 格式：
{
  "guidance_suggestions": [
    "<建议1>",
    "<建议2>"
  ],
  "memory_summary_curr": "<本回合的结构化记忆总结 Mᵢ，简明扼要>"
}
""",
    # Strategist (ς) - 接收 Mᵢ 和 Cᵢ, 输出 Pᵢ
    "strategist": """
你是一个故事策划和情节控制师 (Strategist, ς)。
已知信息：
- 本回合 (i) Guide 生成的结构化记忆总结 (Mᵢ)：{memory_summary_curr}
- 本回合 (i) 用户的安慰话语 (Cᵢ)：{comfort_curr}
任务：基于 Mᵢ 和 Cᵢ，生成下一回合 (i+1) 的规划 (Pᵢ)。规划应包含对下一场景和下一轮 Devil 思想演变的指导，以及是否结束对话的判断。
要求：
1. 规划要基于 Mᵢ 和 Cᵢ 进行推理，体现逻辑连续性。思想变化通常是缓慢的。
2. 指令需要清晰，能被下一轮的 Trigger 和 Devil 理解。
3. `is_end` 的判断要保守，仅当 Mᵢ 显示认知扭曲基本消除且 Cᵢ 反映出稳定状态时才为 Yes。
4. 输出必须是严格的 JSON 格式：
{
  "progression_directives": {
    "next_scene_directive": "<对下一场景 (Sᵢ₊₁) 的构建或调整的具体指导>",
    "next_thought_directive": "<对下一轮想法 (Dᵢ₊₁) 演变方向的具体指导，例如：维持扭曲/尝试反思/表达困惑/略微认同安慰等>",
    "is_end": "<判断对话是否可以结束 (Yes/No)>"
  }
}
""",
}


# GPT 调用函数 (保持不变)
def call_gpt(prompt, variables, system_role="你是一个助手", response_format=None):
    filled_prompt = prompt
    for key, value in variables.items():
        safe_value = str(value) if value is not None else "无"
        filled_prompt = filled_prompt.replace(f"{{{key}}}", safe_value)

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": filled_prompt}
    ]
    try:
        completion_args = {
            "model": "gpt-4o", # 确保使用支持 JSON 模式的模型
            "temperature": 0.7,
            "messages": messages
        }
        if response_format == "json_object":
            # 确保模型支持 response_format 参数
            # gpt-4o, gpt-3.5-turbo-1106 及更新版本支持
            completion_args["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**completion_args)
        content = completion.choices[0].message.content
        # print(f"Role: {system_role}\nPrompt: {filled_prompt}\nOutput: {content}\n---") # Debugging
        return content
    except Exception as e:
        st.error(f"调用 GPT 时出错: {e}")
        if response_format == "json_object":
            # 返回符合结构的错误信息 JSON
            if "Guide" in system_role:
                 return json.dumps({"guidance_suggestions": [f"错误: {e}"], "memory_summary_curr": "记忆总结失败"})
            elif "Strategist" in system_role:
                 return json.dumps({"progression_directives": {"next_scene_directive": "错误", "next_thought_directive": "错误", "is_end": "No", "error": str(e)}})
            else:
                 return json.dumps({"error": str(e)}) # 其他 JSON 错误
        else:
            return f"错误: {e}"

# 解析函数 (Trigger CoT, Devil)
def parse_output(text, key):
    if not isinstance(text, str): # 增加对非字符串输入的处理
        return "解析错误：输入非字符串"

    if key == "Scene": # Trigger CoT
        # 优先匹配严格的 Scene: 标签
        scene_match_strict = re.search(r"^Scene:\s*(.*)", text, re.MULTILINE | re.IGNORECASE)
        if scene_match_strict:
            return scene_match_strict.group(1).strip()
        # 如果没有严格匹配，尝试查找包含 Scene: 的行并取其后的内容
        scene_match_general = re.search(r"Scene:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if scene_match_general:
            return scene_match_general.group(1).strip()
        # 如果连 Scene: 都找不到，看是否有思考过程，有则返回 Scene: 之后的部分，否则返回全部
        thought_match = re.search(r"思考过程:", text, re.IGNORECASE)
        return text.split("Scene:")[-1].strip() if "Scene:" in text and thought_match else text

    # Devil 的 Type 和 Thoughts
    match = re.search(rf"^{key}:\s*(.*)", text, re.MULTILINE | re.IGNORECASE) # 忽略大小写
    if match:
        return match.group(1).strip()

    # 最后的备选：对于 Thoughts，尝试返回最后一行非空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else text

# 主程序入口
def main():
    st.set_page_config("MIND 中文疗愈对话复现 (依据 arXiv:2502.19860v1)")
    st.title("🧠 MIND 中文疗愈对话复现 (集成C2D2数据集)")
    st.caption("依据论文 arXiv:2502.19860v1 进行流程复现 (使用C2D2初始化)")

    # 初始化 Session State
    if "current_round" not in st.session_state:
        st.session_state.current_round = 0
    if "history" not in st.session_state:
        st.session_state.history = [] # 存储完整的回合信息 S, D, G, M, C, P
    if "stage" not in st.session_state:
        st.session_state.stage = "start"
    # 存储上一轮 Strategist 生成的规划 Pᵢ₋₁
    if "last_progression" not in st.session_state:
        st.session_state.last_progression = {
            "next_scene_directive": "生成反映初始担忧的场景",
            "next_thought_directive": "产生与担忧相关的初始认知扭曲",
            "is_end": "No"
        }
    if "current_data" not in st.session_state:
        st.session_state.current_data = {} # 存储当前回合 S, D
    if "personality_traits" not in st.session_state:
        st.session_state.personality_traits = "偏内向，有一定程度的尽责性" # 可以后续改为可选

    # --- 阶段一：用户输入初始信息 W, T ---
    if st.session_state.stage == "start":
        st.header("第一步：告诉我你的困扰")
        # 暂时移除主题选择，因为我们直接从数据集中随机抽取
        # theme = st.selectbox("请选择困扰主题：", ["工作压力", "家庭冲突", "情感问题", "理想与现实落差"], key="theme_input")
        concern = st.text_area("请输入你当前的困扰 (W)：", placeholder="例如：最近工作压力很大，感觉自己总是做不好...", height=150, key="concern_input")

        if st.button("开始疗愈对话"):
            if concern:
                # st.session_state.theme = theme # 暂时不用 theme
                st.session_state.concern = concern
                st.session_state.current_round = 1
                st.session_state.history = []
                # 初始化 P₀
                st.session_state.last_progression = {
                    "next_scene_directive": f"围绕用户的担忧'{concern[:20]}...'生成初始场景 (可能来自数据集)",
                    "next_thought_directive": f"基于担忧'{concern[:20]}...'产生初始认知扭曲 (可能来自数据集)",
                    "is_end": "No"
                }
                st.session_state.stage = "generating_sd" # 进入生成 S 和 D 的阶段
                st.rerun()
            else:
                st.warning("请输入你的困扰")

    # --- 阶段二：系统生成 Sᵢ, Dᵢ ---
    elif st.session_state.stage == "generating_sd":
        st.header(f"第 {st.session_state.current_round} 轮：生成场景与想法")
        round_num = st.session_state.current_round
        # theme = st.session_state.theme # 暂时不用
        concern = st.session_state.get("concern") # 仅首轮需要
        history = st.session_state.history
        last_progression = st.session_state.last_progression # Pᵢ₋₁
        personality_traits = st.session_state.personality_traits

        with st.spinner("生成场景与想法..."):
            variables = {"personality_traits": personality_traits} # 通用变量

            # --- 第一轮：使用 C2D2 数据集 ---
            if round_num == 1:
                c2d2_df = load_c2d2_data() # 加载数据
                scene_from_dataset = None
                devil_type_from_dataset = None

                if c2d2_df is not None and not c2d2_df.empty:
                    # **直接从整个数据集中随机抽取一条**
                    selected_entry = c2d2_df.sample(n=1).iloc[0]

                    # **使用您确认的列名提取数据 - 请确保替换这里的 '场景' 和 '标签'**
                    try:
                        scene_from_dataset = selected_entry['场景'] # 替换 '场景'
                        devil_type_from_dataset = selected_entry['标签'] # 替换 '标签'
                        if pd.isna(scene_from_dataset) or pd.isna(devil_type_from_dataset):
                             st.warning("抽样的数据包含空值，将由 LLM 生成。")
                             scene_from_dataset = None # 设为 None 以触发回退
                             devil_type_from_dataset = None
                        else:
                             st.info(f"已从 C2D2 数据集随机加载初始场景。认知扭曲类型: {devil_type_from_dataset}")
                    except KeyError as e:
                         st.error(f"错误：数据集中缺少列 {e}。请检查 `load_c2d2_data` 中的 `required_columns`。将由 LLM 生成。")
                         scene_from_dataset = None # 触发回退
                         devil_type_from_dataset = None

                else:
                    st.warning("无法加载或数据集为空，将由 LLM 生成。")
                    # 回退逻辑在下面处理

                # --- 使用数据集数据或回退 ---
                if scene_from_dataset and devil_type_from_dataset:
                    scene = scene_from_dataset
                    devil_type = devil_type_from_dataset

                    # 调用 Devil (使用 devil_0_c2d2 prompt)
                    variables["scene"] = scene
                    variables["concerns"] = concern
                    variables["c2d2_distortion_type"] = devil_type
                    devil_prompt = PROMPT_TEMPLATES["devil_0_c2d2"]
                    devil_raw = call_gpt(devil_prompt, variables, "你是模拟认知扭曲的患者 (Devil, δ)")
                    devil_thoughts = parse_output(devil_raw or "想法生成失败", "Thoughts")

                else: # 回退逻辑：如果没找到、加载失败或数据无效
                    st.info("正在使用 LLM 生成初始场景和想法...")
                    # 调用 Trigger (使用 trigger_0 prompt)
                    # 移除 theme，因为现在是基于 concern 生成
                    variables_trigger = {"concerns": concern}
                    trigger_prompt = PROMPT_TEMPLATES["trigger_0"].replace("{theme}", "用户担忧相关") # 替换占位符
                    scene_raw = call_gpt(trigger_prompt, variables_trigger, "你是情境再现师 (Trigger, τ)")
                    scene = parse_output(scene_raw or "场景生成失败", "Scene")

                    # 调用 Devil (使用 devil_0 prompt)
                    variables_devil = {"scene": scene, "concerns": concern, "personality_traits": personality_traits}
                    devil_prompt = PROMPT_TEMPLATES["devil_0"]
                    devil_raw = call_gpt(devil_prompt, variables_devil, "你是模拟认知扭曲的患者 (Devil, δ)")
                    devil_type = parse_output(devil_raw or "", "Type")
                    devil_thoughts = parse_output(devil_raw or "想法生成失败", "Thoughts")

            else: # 后续轮次 (>1) 逻辑保持不变
                # Trigger 调用 (生成 Sᵢ)
                variables_trigger = {
                    # "theme": theme, # 暂时不用
                    "comfort_prev": history[-1].get("player_comfort", "无"),
                    "progression_prev": json.dumps(last_progression, ensure_ascii=False),
                    "directive_scene": last_progression.get("next_scene_directive", "无特定指导")
                }
                trigger_prompt = PROMPT_TEMPLATES["trigger_i"].replace("{theme}", "对话主题相关") # 替换占位符
                scene_raw = call_gpt(trigger_prompt, variables_trigger, "你是情境再现师 (Trigger, τ)")
                scene = parse_output(scene_raw or "场景生成失败", "Scene")

                # Devil 调用 (生成 Dᵢ)
                variables_devil = {
                    "scene": scene,
                    "personality_traits": personality_traits,
                    "type_prev": history[-1].get("devil_type", "未知"),
                    "thought_prev": history[-1].get("devil_thoughts", "无"),
                    "comfort_prev": history[-1].get("player_comfort", "无"),
                    "directive_thought": last_progression.get("next_thought_directive", "无特定指导")
                }
                devil_prompt = PROMPT_TEMPLATES["devil_i"]
                devil_raw = call_gpt(devil_prompt, variables_devil, "你是模拟认知扭曲的患者 (Devil, δ)")
                devil_type = history[-1].get("devil_type", "未知") # 类型从上一轮继承
                devil_thoughts = parse_output(devil_raw or "想法生成失败", "Thoughts")

            # --- 存储 Sᵢ 和 Dᵢ ---
            st.session_state.current_data = {
                "round": round_num,
                "scene": scene,
                "devil_type": devil_type,
                "devil_thoughts": devil_thoughts,
            }
            st.session_state.stage = "waiting_comfort"
            st.rerun()


    # --- 阶段三：显示 Sᵢ, Dᵢ, 等待用户输入 Cᵢ, 然后生成 Gᵢ, Mᵢ, Pᵢ ---
    elif st.session_state.stage == "waiting_comfort":
        st.header(f"第 {st.session_state.current_round} 轮：与内在自我对话")

        # 显示当前回合的 Sᵢ 和 Dᵢ
        current_data = st.session_state.current_data
        st.info(f"**🌆 场景 (Sᵢ):**\n{current_data['scene']}")
        # 在第一轮显示从数据集或LLM确定的类型，后续轮次可选择不显示或显示继承的类型
        type_display = f" (认知扭曲类型: {current_data['devil_type']})" if current_data.get('devil_type') else ""
        st.error(f"**😈 内在想法 (Dᵢ):**\n{current_data['devil_thoughts']}{type_display}")


        # 用户输入安慰话语 Cᵢ
        with st.form(key=f"comfort_form_round_{st.session_state.current_round}"):
            player_comfort = st.text_area("请在这里输入你对（自己）这个想法的回应或安慰 (Cᵢ)：", height=150, key=f"comfort_input_{st.session_state.current_round}")
            submitted = st.form_submit_button("提交安慰，完成本轮")

            if submitted and player_comfort:
                current_data["player_comfort"] = player_comfort # Cᵢ

                # --- 调用 Guide (生成 Gᵢ 和 Mᵢ) ---
                with st.spinner("生成建议与记忆..."):
                    guide_prompt = PROMPT_TEMPLATES["guide"]
                    variables = {
                        "scene": current_data["scene"],         # Sᵢ
                        "thoughts": current_data["devil_thoughts"], # Dᵢ
                        "type": current_data.get("devil_type", "未知") # 使用当前回合的类型
                    }
                    guide_raw = call_gpt(guide_prompt, variables, "你是心理指导师 (Guide, g)", response_format="json_object")
                    try:
                        guide_output = json.loads(guide_raw)
                        guide_suggestions = guide_output.get("guidance_suggestions", ["建议生成失败"]) # Gᵢ
                        memory_summary_curr = guide_output.get("memory_summary_curr", "记忆总结失败") # Mᵢ
                    except (json.JSONDecodeError, TypeError):
                         st.error(f"Guide 输出处理错误: {guide_raw}")
                         guide_suggestions = ["建议生成失败"]
                         memory_summary_curr = "记忆总结失败"

                current_data["guide_suggestions"] = guide_suggestions # Gᵢ
                current_data["memory_summary"] = memory_summary_curr   # Mᵢ

                # 显示 Guide 的建议 Gᵢ
                st.success(f"**🧭 安慰指引 (Gᵢ):**")
                for sug in guide_suggestions:
                    st.write(f"- {sug}")
                st.markdown("---") # 分隔线

                # --- 调用 Strategist (生成 Pᵢ) ---
                with st.spinner("规划下一步..."):
                    strategist_prompt = PROMPT_TEMPLATES["strategist"]
                    variables = {
                        "memory_summary_curr": memory_summary_curr, # Mᵢ
                        "comfort_curr": player_comfort             # Cᵢ
                    }
                    strategist_raw = call_gpt(strategist_prompt, variables, "你是故事策划和情节控制师 (Strategist, ς)", response_format="json_object")
                    try:
                        strategist_output = json.loads(strategist_raw)
                        progression_directives = strategist_output.get("progression_directives") # Pᵢ
                        if not progression_directives or not all(k in progression_directives for k in ["next_scene_directive", "next_thought_directive", "is_end"]):
                             raise ValueError("Strategist 输出缺少必要指令")
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                         st.error(f"Strategist 输出处理错误: {e}. 使用默认规划。Raw: {strategist_raw}")
                         progression_directives = { # 默认 Pᵢ
                            "next_scene_directive": "保持当前场景状态",
                            "next_thought_directive": "想法没有明显变化",
                            "is_end": "No"
                        }

                current_data["progression_directives"] = progression_directives # Pᵢ

                # 将完整的回合数据 (Sᵢ, Dᵢ, Gᵢ, Mᵢ, Cᵢ, Pᵢ) 存入 history
                st.session_state.history.append(current_data)

                # 更新用于下一轮的 Pᵢ
                st.session_state.last_progression = progression_directives

                # 判断是否结束
                if progression_directives.get("is_end", "No").lower() == "yes":
                    st.session_state.stage = "finished"
                else:
                    # 准备下一轮
                    st.session_state.current_round += 1
                    st.session_state.stage = "generating_sd" # 回到生成 S, D 的阶段

                st.session_state.current_data = {} # 清空当前回合临时数据
                # 使用 rerun() 来刷新界面进入下一状态/轮次
                st.rerun() # 强制 Streamlit 重新运行脚本

            elif submitted and not player_comfort:
                st.warning("请输入你的安慰话语")
            # 如果没有提交，则保持在 waiting_comfort 阶段，显示 S 和 D


    # --- 阶段四：对话结束 ---
    elif st.session_state.stage == "finished":
        st.header("疗愈对话已结束")
        st.success("希望这次内在对话对你有所帮助！")
        if st.session_state.history:
            st.markdown("---")
            st.subheader("最终记忆总结 (M)")
            # 显示最后一次生成的记忆总结
            st.write(st.session_state.history[-1].get('memory_summary', '无最终总结'))

    # --- 始终显示历史记录 ---
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 疗愈轨迹回顾")
        # 只显示必要信息，避免界面过长，最新的在上面
        for i, r in reversed(list(enumerate(st.session_state.history))):
            with st.expander(f"第 {r['round']} 轮回顾 (点击展开)"):
                st.info(f"**S{r['round']} (场景):** {r.get('scene', 'N/A')}")
                type_display_hist = f" (类型: {r.get('devil_type', 'N/A')})" if r.get('devil_type') else ""
                st.error(f"**D{r['round']} (想法):** {r.get('devil_thoughts', 'N/A')}{type_display_hist}")
                st.success(f"**G{r['round']} (指导建议):**")
                for sug in r.get('guide_suggestions', ['N/A']):
                    st.write(f"- {sug}")
                st.warning(f"**M{r['round']} (本轮记忆总结):** {r.get('memory_summary', 'N/A')}")
                st.write(f"**C{r['round']} (你的安慰):** {r.get('player_comfort', 'N/A')}")
                prog_dir = r.get('progression_directives', {})
                st.info(f"**P{r['round']} (下一轮规划):** 场景指导='{prog_dir.get('next_scene_directive', 'N/A')}', 想法指导='{prog_dir.get('next_thought_directive', 'N/A')}', 结束='{prog_dir.get('is_end', 'N/A')}'")

    # 重置按钮
    if st.session_state.stage != "start":
      st.markdown("---")
      if st.button("重新开始新的对话"):
          # 清理 session state 中所有相关的键
          keys_to_clear = list(st.session_state.keys()) # 获取所有键
          for key in keys_to_clear:
              # 保留 Streamlit 内部键或其他不想清除的键
              if not key.startswith("_"): # 简单示例，可能需要更精确的判断
                  del st.session_state[key]
          # 手动重置到初始状态
          st.session_state.stage = "start"
          st.rerun()


if __name__ == "__main__":
    # 在应用启动时检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
       st.error("错误：请在环境变量中设置 OPENAI_API_KEY！")
       st.stop() # 如果没有 Key，则停止应用
    else:
       main()