# 触发部署
from openai import OpenAI
import streamlit as st
import matplotlib.pyplot as plt
import os

client = OpenAI(api_key="")

# GPT 调用函数
def call_gpt(role_prompt, user_input, system_role="你是一个心理角色"):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": role_prompt.replace("{{input}}", user_input)}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[错误] {str(e)}"

# 情感演化计算
def calculate_emotion(devil, player):
    devil_score = devil.count("我") * 2
    player_score = player.count("我") * 3
    emotion_score = max(0, min(100, 50 + player_score - devil_score))
    return emotion_score

# 绘图函数（去除字体设置，适配部署）
def plot_emotion_trajectory():
    emotion_scores = [round["emotion_score"] for round in st.session_state["rounds"]]
    plt.figure(figsize=(10, 5))
    plt.plot(emotion_scores, marker='o', linestyle='-', color='b')
    plt.title("情感转化轨迹", fontsize=16)
    plt.xlabel("疗愈轮次", fontsize=12)
    plt.ylabel("情感评分", fontsize=12)
    plt.xticks(range(len(emotion_scores)))
    plt.ylim(0, 100)
    plt.grid(True)
    st.pyplot(plt)

# 自我融合总结生成函数
def generate_summary():
    st.markdown("## 🪞 自我融合总结")

    all_devil = "\n".join([r["devil"] for r in st.session_state["rounds"]])
    all_guide = "\n".join([r["guide"] for r in st.session_state["rounds"]])
    all_player = "\n".join([r["player"] for r in st.session_state["rounds"]])

    summary_prompt = f"""
以下是你内心的多轮心理对话内容：

【认知扭曲合集】
{all_devil}

【心理建议合集】
{all_guide}

【自我安慰合集】
{all_player}

请你作为“Player角色”，用第一人称总结你的内在旅程：
- 你有哪些觉察或改变？
- 有哪些感受得到了安慰？
- 你如何理解现在的自己？
请生成一段不少于100字、温柔内省的疗愈总结。
"""

    result = call_gpt(summary_prompt, "", "你是一个觉察自我情绪、接纳成长的角色")
    st.success("🎉 自我总结已生成：")
    st.write(result)

# 主程序入口
def main():
    st.set_page_config(page_title="MIND 中文心理疗愈", layout="centered")
    st.title("🧠 MIND 中文疗愈对话体验")
    st.markdown("<br/>", unsafe_allow_html=True)

    if "rounds" not in st.session_state:
        st.session_state["rounds"] = []

    theme = st.selectbox("请选择困扰主题：", ["工作压力", "家庭冲突", "情感问题", "理想与现实落差"])
    concern = st.text_area("请输入你当前的困扰：", height=150)

    if st.button("继续疗愈对话") and concern:
        with st.spinner("正在生成疗愈内容，请稍候..."):

            previous = st.session_state["rounds"][-1] if st.session_state["rounds"] else None
            memory_text = f"上一轮场景：{previous['scene']}；认知扭曲：{previous['devil']}；引导建议：{previous['guide']}；自我安慰：{previous['player']}" if previous else ""

            trigger_prompt = f"""
你是一个情境创作者。当前主题：{theme}，当前困扰：{{input}}。
{memory_text}
请生成一个贴近现实、容易代入的生活片段，控制在150字以内，不要加入分析。
"""
            scene = call_gpt(trigger_prompt, concern, "你是一个独立的、富有画面感的叙事设计人格。")

            devil_prompt = """
以下是一个生活场景：{{input}}
请模拟你内心中最批判、否定的声音。你倾向于否定自我、以偏概全、极端思维。
用第一人称，说出1~2条消极想法（每条不超10字）。
"""
            devil = call_gpt(devil_prompt, scene, "你是角色内心的认知扭曲人格，自我否定强烈，语言短促极端。")

            guide_prompt = """
以下是场景与内心想法：{{input}}
请用温柔、非评判、结构化的方式，给出1~2条认知重构建议（每条不超15字）。
"""
            guide_input = f"场景：{scene}\n认知扭曲：{devil}"
            guide = call_gpt(guide_prompt, guide_input, "你是角色内在的心理咨询人格，语言理智温和，有认知行为疗法风格。")

            player_prompt = """
以下是认知建议：{{input}}
请模拟你内心真实的声音，表达你如何接纳自己、鼓励自己。
输出不超过2段温柔安慰语句，用第一人称。
"""
            comfort = call_gpt(player_prompt, guide, "你是角色内在的自我调节人格，语言真诚、感性、基于个人经验。")

            emotion_score = calculate_emotion(devil, comfort)

            st.session_state["rounds"].append({
                "scene": scene,
                "devil": devil,
                "guide": guide,
                "player": comfort,
                "emotion_score": emotion_score
            })

    if st.session_state["rounds"]:
        st.markdown("## 📜 疗愈轨迹回顾")
        for i, round in enumerate(st.session_state["rounds"]):
            st.markdown(f"### 第 {i+1} 轮")
            st.info(f"🌆 场景：{round['scene']}")
            st.error(f"😈 认知扭曲：{round['devil']}")
            st.success(f"🧭 引导建议：{round['guide']}")
            st.write(f"💬 自我安慰：{round['player']}")
            st.write(f"🎯 情感评分：{round['emotion_score']} / 100")
        plot_emotion_trajectory()

        if st.button("🪞 生成自我融合总结"):
            generate_summary()

if __name__ == "__main__":
    main()
