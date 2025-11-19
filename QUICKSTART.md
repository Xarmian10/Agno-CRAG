# 快速开始指南（5 分钟上手）

## 📋 前置准备

1. **Python 3.10+** 已安装
2. **SiliconFlow API Key** ([获取地址](https://siliconflow.cn))

## 🚀 5 步启动

### 步骤 1: 安装依赖 (2 分钟)

```bash
# 克隆项目
git clone <your-repo-url>
cd Agno-RAG

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 配置 API Key (1 分钟)

创建 `.env` 文件：

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

编辑 `.env`，添加您的 API Key：

```bash
SILICONFLOW_API_KEY=your_api_key_here
```

### 步骤 3: 启动服务 (30 秒)

```bash
python agno_agent.py
```

看到这个就成功了：

```
访问地址: http://127.0.0.1:7777
```

### 步骤 4: 上传文档 (1 分钟)

打开浏览器访问 http://127.0.0.1:7777

在对话框输入：

```
上传 pdf/GB+146.1-2020.pdf
```

或使用脚本批量上传：

```bash
python upload_documents.py --dir pdf
```

### 步骤 5: 开始提问！(30 秒)

在对话框输入：

```
GB146 标准中关于客车的安全要求是什么？
```

等待几秒，系统会：
1. 🔍 检索相关文档
2. 🧠 CRAG 评估质量
3. ✂️ 精炼知识片段
4. 💬 生成答案

---

## 🎯 完成！

现在您已经有了一个功能完整的 CRAG 问答系统！

## 🔍 查看日志

终端会显示详细的检索和 CRAG 评估过程：

```
============================================================
开始查询文档
============================================================
查询: GB146 标准中关于客车的安全要求是什么？

[检索阶段 1/5] 向量搜索...
检索到文档数: 10

CRAG评估模式: 完整CRAG
  [完成] 决策动作: correct
  最高文档分数: 0.8234

[完成] 检索完成 - 总耗时: 0.89秒
============================================================
```

## 📚 下一步

- 📖 阅读 [README.md](README.md) 了解详细功能
- ⚙️ 查看 [env.example](env.example) 调整配置
- 🎓 了解 [CRAG 原理](README.md#-crag-原理详解)

## ❓ 遇到问题？

### 问题 1: 模型下载慢

**解决**：使用国内镜像

```bash
# 在 .env 中添加
HF_ENDPOINT=https://hf-mirror.com
```

### 问题 2: 找不到 PDF 文件

**解决**：使用绝对路径

```bash
# Windows
上传 C:\Users\YourName\Documents\document.pdf

# Linux/Mac
上传 /home/yourname/documents/document.pdf
```

### 问题 3: API Key 无效

**解决**：
1. 检查 `.env` 文件中的 API Key 是否正确
2. 确保没有多余的空格或引号
3. 重启服务

---

## 🎉 享受您的智能文档问答系统！

有问题？查看 [README.md](README.md) 的常见问题部分。

