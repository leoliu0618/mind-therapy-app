import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from openai import OpenAI

# 初始化 OpenAI 客户端（使用环境变量管理 API Key）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


# 情感演化评分函数
def calculate_emotion(devil, player):
    devil_score = devil.count("我") * 2
    player_score = player.count("我") * 3
    emotion_score = max(0, min(100, 50 + player_score - devil_score))
    return emotion_score


# 绘制情感评分轨迹（含平滑）
def plot_emotion_trajectory():
    scores = [round["emotion_score"] for round in st.session_state["rounds"]]
    smooth_scores = gaussian_filter1d(scores, sigma=1)

    plt.figure(figsize=(10, 5))
    plt.plot(scores, 'bo-', label='原始分数')
    plt.plot(smooth_scores, 'r--', label='平滑曲线')
    plt.title("情感评分轨迹")
    plt.xlabel("轮次")
    plt.ylabel("评分")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)


# 生成自我融合总结
def generate_summary():
    st.markdown("## 🪞 自我融合总结")
    all_devil = "\n".join([r["devil"] for r in st.session_state["rounds"]])
    all_guide = "\n".join([r["guide"] for r in st.session_state["rounds"]])
    all_player = "\n".join([r["player"] for r in st.session_state["rounds"]])
    summary_prompt = f"""
以下是你内心的多轮心理对话内容：

【认知扭曲合集】\n{all_devil}
【心理建议合集】\n{all_guide}
【自我安慰合集】\n{all_player}

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
    st.set_page_config("MIND 中文疗愈对话体验")
    st.title("🧠 MIND 中文疗愈对话体验")

    if "rounds" not in st.session_state:
        st.session_state["rounds"] = []
    if "theme_memory" not in st.session_state:
        st.session_state["theme_memory"] = {}

    theme = st.selectbox("请选择困扰主题：", ["工作压力", "家庭冲突", "情感问题", "理想与现实落差"])
    concern = st.text_area("请输入你当前的困扰：", height=150)

    if theme not in st.session_state["theme_memory"]:
        st.session_state["theme_memory"][theme] = []

    if st.button("继续疗愈对话") and concern:
        with st.spinner("正在生成疗愈内容..."):
            previous = st.session_state["theme_memory"][theme][-1] if st.session_state["theme_memory"][theme] else None
            memory_text = f"上一轮：场景={previous['scene']}；扭曲={previous['devil']}；建议={previous['guide']}；安慰={previous['player']}" if previous else ""

            trigger_prompt = f"""
你是情境创作者。当前主题：{theme}，当前困扰：{{input}}。{memory_text}
请生成一个贴近现实的生活场景，不超过150字。
"""
            scene = call_gpt(trigger_prompt, concern, "你是富有画面感的叙事设计人格。")

            devil = call_gpt("""
以下是场景：{{input}}
模拟内心最批判的声音，用第一人称，说出1~2条否定想法，每条不超10字。
""", scene, "你是认知扭曲人格。")

            guide_input = f"场景：{scene}\n扭曲：{devil}"
            guide = call_gpt("""
以下内容：{{input}}
请生成1~2条认知重构建议，每条不超15字。
""", guide_input, "你是理性温和的心理咨询人格。")

            strategist = call_gpt("""
以下内容：{{input}}
请作为策略师，从元认知角度分析该认知偏差成因，并给出1条结构性建议，不超过80字。
""", guide_input, "你是理性结构化的认知策略师人格。")

            comfort = call_gpt("""
以下为认知建议：{{input}}
请模拟安慰性声音，用第一人称表达情绪接纳与鼓励。
""", guide, "你是温柔真诚的自我调节人格。")

            score = calculate_emotion(devil, comfort)

            st.session_state["theme_memory"][theme].append({
                "scene": scene,
                "devil": devil,
                "guide": guide,
                "strategist": strategist,
                "player": comfort,
                "emotion_score": score
            })

    # 展示所有轮次
    rounds = st.session_state["theme_memory"][theme]
    if rounds:
        st.markdown("## 📜 疗愈轨迹回顾")
        for i, r in enumerate(rounds):
            with st.expander(f"第{i + 1}轮 🎯 情感评分：{r['emotion_score']} / 100"):
                st.info(f"🌆 场景：{r['scene']}")
                st.error(f"😈 认知扭曲：{r['devil']}")
                st.success(f"🧭 引导建议：{r['guide']}")
                st.warning(f"🧠 拆解思考：{r['strategist']}")
                st.write(f"💬 自我安慰：{r['player']}")

        plot_emotion_trajectory()

        if st.button("🪞 生成自我融合总结"):
            generate_summary()


if __name__ == "__main__":
    main()
