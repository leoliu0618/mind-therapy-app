import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from openai import OpenAI
import json # 引入json库方便处理结构化输出

# 初始化 OpenAI 客户端（使用环境变量管理 API Key）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Prompt Templates (更贴近论文附录 C) ---
# 注意：实际使用时需要根据上下文变量填充 {var}
# 这些是根据附录 C 简化和翻译的示例，实际效果依赖于 Prompt Engineering
PROMPT_TEMPLATES = {
    "trigger_0": """
你是一个情景再现师。你需要以{theme}为主题生成一个模拟场景：包括角色互动、场景描述，制造糟糕的情境和矛盾。
患者具有“{concerns}”的烦恼，这种烦恼表现了患者的认知扭曲。
要求：
1. 场景应充分反映患者的状态、体现出患者的烦恼。
2. 场景是故事的背景部分，不包含对话和心理描述。
3. 不包含价值判断。
4. 输出格式：
Scene: <生成的模拟场景，不超过150字>
""",
    "trigger_i": """
你是一个情景再现师。你需要以{theme}为主题，基于上一轮用户的安慰 {comfort_prev} 和策略师的指导 {progression_prev}，对历史场景进行扩展或保持不变。
历史场景概要：{memory_summary}
要求：
1. 扩展要符合发展逻辑，与历史场景契合。
2. 场景是故事的背景部分，不包含对话和心理描述。
3. 不包含价值判断。
4. 输出格式：
Scene: <生成的模拟场景，不超过150字>
""",
    "devil_0": """
你是一个正在经历认知扭曲的患者。
情境：{scene}
你的烦恼是：{concerns}
请模拟第一人称视角，产生一个核心的负面想法（认知扭曲），并说明其类型（例如：过度概括、非此即彼、情绪化推理等）。
要求：
1. 想法要符合情境和烦恼。
2. 简短，像内心闪过的念头。
3. 输出格式：
Type: <认知扭曲类型>
Thoughts: <第一人称的想法，不超过20字>
""",
    "devil_i": """
你是一个正在经历认知扭曲的患者。
情境：{scene}
你的认知扭曲类型大致是：{type_prev}
上一轮你的想法是：{thought_prev}
上一轮安慰者的话是：{comfort_prev}
上一轮策略师判定你的思想变化方向是：{progression_prev}
请根据当前情境、上一轮的互动和策略师的指导，模拟你此刻第一人称可能的想法。这个想法可能是对安慰的回应（肯定或反驳），并可能体现出思想上的微小变化或固守。
要求：
1. 想法要符合情境和互动历史。
2. 简短，像内心闪过的念头。
3. 输出格式：
Thoughts: <第一人称的想法，不超过20字>
""",
    "guide": """
你是一个专业的心理指导师。
当前情景：{scene}
患者的认知扭曲想法：{thoughts} (类型: {type})
你的任务是：针对上述情景和想法，给出1-2条具体的、可操作的安慰引导建议，帮助“安慰者”进行认知重构。
要求：
1. 建议要紧密结合情景和患者想法。
2. 建议是给“安慰者”的，指导其如何安慰。
3. 简短精炼，每条不超过20字。
4. 输出格式：
Suggestions:
1. <建议1>
2. <建议2>
""",
    "strategist": """
你是一个故事策划和情节控制师。
已知信息：
- 当前回合用户的安慰话语：{comfort_curr}
- 本回合场景：{scene}
- 本回合患者想法：{thoughts}
- 历史对话概要：{memory_summary}
你的任务是：
1. 判断基于用户的安慰，患者（Devil）的思想是否应发生转变。
2. 设计下一回合故事的基本走向（或保持不变）。
3. 描述下一回合患者思想可能的转变方向（或保持不变）。
4. 判断对话是否可以结束 (Is_end: Yes/No)。
要求：
1. 思想变化是缓慢、合理的，通常需要积极有效的安慰才会发生。
2. 决策要符合逻辑连续性。
3. 输出格式必须是 JSON:
{
  "next_scene_idea": "<下一场景的核心思路或维持现状>",
  "next_thought_direction": "<下一轮想法的转变方向或维持现状>",
  "is_end": "<Yes/No>"
}
""",
}


# GPT 调用函数 (增加 JSON 解析)
def call_gpt(prompt, variables, system_role="你是一个助手", response_format=None):
    filled_prompt = prompt
    for key, value in variables.items():
        filled_prompt = filled_prompt.replace(f"{{{key}}}", str(value))

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": filled_prompt}
    ]
    try:
        if response_format == "json_object":
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=messages
            )
        else:
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=messages
            )
        content = completion.choices[0].message.content
        # print(f"Role: {system_role}\nPrompt: {filled_prompt}\nOutput: {content}\n---") # Debugging
        return content
    except Exception as e:
        st.error(f"调用 GPT 时出错: {e}")
        return None

# 解析特定格式的输出
def parse_output(text, key):
    lines = text.split('\n')
    for line in lines:
        if line.startswith(key + ":"):
            return line[len(key)+1:].strip()
    # 如果找不到特定 key，尝试返回第一行或全部内容作为备选
    return lines[0].strip() if lines else text

def parse_suggestions(text):
    lines = text.split('\n')
    suggestions = []
    parsing = False
    for line in lines:
        if line.strip().startswith("Suggestions:"):
            parsing = True
            continue
        if parsing and line.strip():
            # 移除可能的 "1.", "2." 等前缀
            parts = line.strip().split('.', 1)
            if len(parts) > 1 and parts[0].isdigit():
                suggestions.append(parts[1].strip())
            else:
                suggestions.append(line.strip())
    # 如果解析失败，返回原始文本
    return suggestions if suggestions else [text.strip()]


# 主程序入口
def main():
    st.set_page_config("MIND 中文疗愈对话复现 (依据 arXiv:2502.19860v1)")
    st.title("🧠 MIND 中文疗愈对话复现")
    st.caption("依据论文 arXiv:2502.19860v1 进行流程复现")

    # 初始化 Session State
    if "current_round" not in st.session_state:
        st.session_state["current_round"] = 0
    if "history" not in st.session_state:
        st.session_state["history"] = [] # 存储完整的回合信息
    if "stage" not in st.session_state:
        st.session_state["stage"] = "start" # 控制流程阶段: start, generated, waiting_comfort, finished
    if "last_strategist_output" not in st.session_state:
        st.session_state["last_strategist_output"] = {"next_scene_idea": "初始场景", "next_thought_direction": "初始想法", "is_end": "No"}
    if "current_data" not in st.session_state:
        st.session_state["current_data"] = {} # 存储当前回合生成的数据 S, D, G


    # --- 阶段一：用户输入初始信息 ---
    if st.session_state["stage"] == "start":
        st.header("第一步：告诉我你的困扰")
        theme = st.selectbox("请选择困扰主题：", ["工作压力", "家庭冲突", "情感问题", "理想与现实落差"], key="theme_input")
        concern = st.text_area("请输入你当前的困扰：", height=150, key="concern_input")

        if st.button("开始疗愈对话"):
            if concern:
                st.session_state["theme"] = theme
                st.session_state["concern"] = concern
                st.session_state["current_round"] = 1
                st.session_state["history"] = []
                st.session_state["last_strategist_output"] = {"next_scene_idea": "初始场景", "next_thought_direction": "初始想法", "is_end": "No"}
                st.session_state["stage"] = "generated"
                st.rerun() # 重新运行以进入下一阶段
            else:
                st.warning("请输入你的困扰")

    # --- 阶段二：系统生成 S, D, G 并等待用户输入 C ---
    elif st.session_state["stage"] == "generated":
        st.header(f"第 {st.session_state['current_round']} 轮：与内在自我对话")
        round_num = st.session_state["current_round"]
        theme = st.session_state["theme"]
        concern = st.session_state["concern"] # 首轮需要，后续理论上不需要
        history = st.session_state["history"]
        last_strategist_output = st.session_state["last_strategist_output"]

        with st.spinner("生成场景、想法与建议..."):
            # 准备输入变量
            variables = {"theme": theme}
            # Trigger 输入
            if round_num == 1:
                trigger_prompt = PROMPT_TEMPLATES["trigger_0"]
                variables["concerns"] = concern
            else:
                trigger_prompt = PROMPT_TEMPLATES["trigger_i"]
                variables["comfort_prev"] = history[-1]["player_comfort"]
                variables["progression_prev"] = last_strategist_output.get("next_scene_idea", "无特定指导")
                variables["memory_summary"] = f"之前场景关于: {history[-1]['scene'][:30]}... 想法倾向: {history[-1]['devil_thoughts'][:20]}..." # 简易记忆

            scene_raw = call_gpt(trigger_prompt, variables, "你是情境创作者")
            scene = parse_output(scene_raw or "", "Scene")

            # Devil 输入
            variables = {"scene": scene}
            if round_num == 1:
                devil_prompt = PROMPT_TEMPLATES["devil_0"]
                variables["concerns"] = concern
            else:
                devil_prompt = PROMPT_TEMPLATES["devil_i"]
                variables["type_prev"] = history[-1].get("devil_type", "未知")
                variables["thought_prev"] = history[-1]["devil_thoughts"]
                variables["comfort_prev"] = history[-1]["player_comfort"]
                variables["progression_prev"] = last_strategist_output.get("next_thought_direction", "无特定指导")

            devil_raw = call_gpt(devil_prompt, variables, "你是认知扭曲人格")
            devil_type = parse_output(devil_raw or "", "Type") if round_num == 1 else history[-1].get("devil_type", "未知") # 类型只在第一轮识别
            devil_thoughts = parse_output(devil_raw or "", "Thoughts")

            # Guide 输入
            guide_prompt = PROMPT_TEMPLATES["guide"]
            variables = {"scene": scene, "thoughts": devil_thoughts, "type": devil_type}
            guide_raw = call_gpt(guide_prompt, variables, "你是心理指导师")
            guide_suggestions = parse_suggestions(guide_raw or "")

        # 存储当前生成的数据，等待用户输入
        st.session_state["current_data"] = {
            "round": round_num,
            "scene": scene,
            "devil_type": devil_type,
            "devil_thoughts": devil_thoughts,
            "guide_suggestions": guide_suggestions,
        }

        # 显示给用户 S, D, G
        st.info(f"**🌆 场景 (Scene):**\n{scene}")
        st.error(f"**😈 内在想法 (Devil's Thoughts):**\n{devil_thoughts} {(f' (类型: {devil_type})' if round_num == 1 else '')}")
        st.success("**🧭 安慰指引 (Guide's Suggestions):**")
        for sug in guide_suggestions:
            st.write(f"- {sug}")

        # 进入等待用户输入的阶段
        st.session_state["stage"] = "waiting_comfort"
        st.rerun() # 重新运行以显示输入框


    # --- 阶段三：用户输入安慰 C，系统生成 P 并完成本轮 ---
    elif st.session_state["stage"] == "waiting_comfort":
        st.header(f"第 {st.session_state['current_round']} 轮：输入你的安慰")

        # 显示上一阶段生成的信息
        current_data = st.session_state["current_data"]
        st.info(f"**🌆 场景 (Scene):**\n{current_data['scene']}")
        st.error(f"**😈 内在想法 (Devil's Thoughts):**\n{current_data['devil_thoughts']} {(f' (类型: {current_data['devil_type']})' if current_data['round'] == 1 else '')}")
        st.success("**🧭 安慰指引 (Guide's Suggestions):**")
        for sug in current_data['guide_suggestions']:
            st.write(f"- {sug}")

        # 用户输入安慰话语 C_i
        with st.form(key=f"comfort_form_round_{st.session_state['current_round']}"):
            player_comfort = st.text_area("请在这里输入你对（自己）这个想法的回应或安慰：", height=150, key=f"comfort_input_{st.session_state['current_round']}")
            submitted = st.form_submit_button("提交安慰，继续下一轮")

            if submitted and player_comfort:
                current_data["player_comfort"] = player_comfort

                # 调用 Strategist
                with st.spinner("思考下一步..."):
                    strategist_prompt = PROMPT_TEMPLATES["strategist"]
                    # 构建简易记忆给 Strategist
                    memory_summary = "\n".join([f"R{r['round']}: Scene={r['scene'][:20]}..., Thought={r['devil_thoughts'][:15]}..., Comfort={r['player_comfort'][:15]}..." for r in st.session_state["history"]])

                    variables = {
                        "comfort_curr": player_comfort,
                        "scene": current_data["scene"],
                        "thoughts": current_data["devil_thoughts"],
                        "memory_summary": memory_summary if memory_summary else "无历史记录"
                    }
                    strategist_raw = call_gpt(strategist_prompt, variables, "你是认知策略师", response_format="json_object")

                    try:
                        strategist_output = json.loads(strategist_raw) if strategist_raw else {"next_scene_idea": "保持现状", "next_thought_direction": "保持现状", "is_end": "No"}
                    except json.JSONDecodeError:
                        st.warning(f"Strategist 输出格式错误，使用默认值: {strategist_raw}")
                        strategist_output = {"next_scene_idea": "保持现状", "next_thought_direction": "保持现状", "is_end": "No"}

                current_data["strategist_output"] = strategist_output

                # 将完整的回合数据存入 history
                st.session_state["history"].append(current_data)
                st.session_state["last_strategist_output"] = strategist_output # 更新给下一轮使用

                # 判断是否结束
                if strategist_output.get("is_end", "No").lower() == "yes":
                    st.session_state["stage"] = "finished"
                else:
                    # 准备下一轮
                    st.session_state["current_round"] += 1
                    st.session_state["stage"] = "generated"

                st.session_state["current_data"] = {} # 清空当前数据
                st.rerun() # 进入下一轮或结束

            elif submitted and not player_comfort:
                st.warning("请输入你的安慰话语")


    # --- 阶段四：对话结束 ---
    elif st.session_state["stage"] == "finished":
        st.header("疗愈对话已结束")
        st.success("希望这次内在对话对你有所帮助！")
        # 可以选择在这里自动生成总结


    # --- 始终显示历史记录 ---
    if st.session_state["history"]:
        st.markdown("---")
        st.subheader("📜 疗愈轨迹回顾")
        for i, r in enumerate(st.session_state["history"]):
            with st.expander(f"第 {r['round']} 轮回顾"):
                st.info(f"**🌆 场景:** {r['scene']}")
                st.error(f"**😈 内在想法:** {r['devil_thoughts']} {(f' (类型: {r['devil_type']})' if r['round'] == 1 else '')}")
                st.success("**🧭 安慰指引:**")
                for sug in r['guide_suggestions']:
                    st.write(f"- {sug}")
                st.write(f"**💬 你的安慰:** {r['player_comfort']}")
                # 显示 Strategist 的决策供参考
                strat_out = r.get('strategist_output', {})
                st.warning(f"**🧠 策略师决策:** 下一场景思路='{strat_out.get('next_scene_idea', 'N/A')}', 下一想法方向='{strat_out.get('next_thought_direction', 'N/A')}', 结束='{strat_out.get('is_end', 'N/A')}'")

        # 可以在这里保留或移除情感评分和总结功能
        # plot_emotion_trajectory() # 情感评分需要调整逻辑或移除
        # if st.button("🪞 生成自我融合总结"):
        #     generate_summary() # 总结功能可以保留

    # 添加一个重置按钮，方便重新开始
    if st.session_state["stage"] != "start":
      if st.button("重新开始新的对话"):
          # 清理 session state
          keys_to_reset = ["current_round", "history", "stage", "last_strategist_output", "current_data", "theme", "concern"]
          for key in keys_to_reset:
              if key in st.session_state:
                  del st.session_state[key]
          st.rerun()


if __name__ == "__main__":
    # 注意：运行前请确保设置了 OPENAI_API_KEY 环境变量
    if not os.getenv("OPENAI_API_KEY"):
       st.error("错误：请设置 OPENAI_API_KEY 环境变量！")
    else:
       main()
