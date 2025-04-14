# 🧠 MIND 中文疗愈对话复现（集成 C2D2 数据集）

本项目基于论文 [MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogue (arXiv:2502.19860v1)](https://arxiv.org/abs/2502.19860) 实现中文疗愈对话系统，复现其多轮自我对话流程（Strategist、Guide 等角色）并引入 [C2D2 数据集](https://huggingface.co/datasets/IMCS-DQA/C2D2) 进行场景初始化，增强疗愈语境真实感。

---

## ✨ 特性功能

- ✅ 多轮中文疗愈对话流（Trigger → Strategist → Guide → Critic 等角色协作）
- ✅ 支持用户输入中文困扰（如工作压力、人际关系等）
- ✅ 使用 C2D2 数据集进行首轮场景生成，增强语境真实感
- ✅ 部署于 [Streamlit](https://streamlit.io/) Cloud，无需本地安装即可体验

---

## 🚀 在线体验

👉 [点击访问应用](https://leo.streamlit.app)（或使用你部署后的实际链接）

---


## 🛠️ 环境依赖

所有依赖已整理于 `requirements.txt`，包含主要依赖如下：

- `streamlit==1.44.1`
- `openai==1.72.0`
- `pandas`, `numpy`, `matplotlib`
- `scipy`, `python-dotenv`
- `GitPython`, `pydeck`, `httpx`, `altair` 等

完整环境一键安装：

```bash
pip install -r requirements.txt


📄 License
本项目遵循 MIT License

📬 联系开发者
欢迎联系作者：leo@10000.life
或通过 Issues 提交建议与反馈！

🙋‍♂️ 致谢与参考
原论文作者团队：MIND 论文提出者

HuggingFace 社区 C2D2 数据集

Streamlit Cloud 部署平台



