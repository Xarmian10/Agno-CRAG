# Agno-RAG：基于 CRAG 的智能文档问答系统

<div align="center">

**一个实现了 CRAG（Corrective Retrieval Augmented Generation）论文的高级 RAG 系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Agno](https://img.shields.io/badge/Agno-Framework-green.svg)](https://agno.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[功能特点](#-功能特点) • [快速开始](#-快速开始) • [项目架构](#-项目架构) • [CRAG 原理](#-crag-原理) • [使用指南](#-使用指南) • [配置说明](#-配置说明)

</div>

---

## 📖 项目简介

Agno-RAG 是一个基于 **CRAG（Corrective Retrieval Augmented Generation）** 论文实现的高级 RAG 系统。与传统 RAG 系统不同，CRAG 能够自动评估检索质量，并根据评估结果采取不同的纠正策略，显著提高了问答准确性和鲁棒性。

### 什么是 RAG？

**RAG（Retrieval-Augmented Generation）** = 检索增强生成

传统 LLM 的问题：
- ❌ 知识固化（训练后无法更新）
- ❌ 容易"幻觉"（编造不存在的信息）
- ❌ 无法访问私有/专业文档

RAG 的解决方案：
- ✅ 从外部知识库检索相关文档
- ✅ 将检索结果作为上下文提供给 LLM
- ✅ LLM 基于真实文档回答问题

### 什么是 CRAG？

**CRAG（Corrective Retrieval Augmented Generation）** = 纠正式检索增强生成

传统 RAG 的问题：
- ❌ 无法判断检索质量（如果检索错误怎么办？）
- ❌ 盲目使用所有检索结果（包含大量无关信息）
- ❌ 缺少纠正机制

CRAG 的创新：
- ✅ **智能评估**：自动评估每个检索文档的相关性
- ✅ **动作路由**：根据评估结果采取不同策略
  - **Correct**：检索质量高 → 精炼知识片段
  - **Incorrect**：检索质量低 → 使用 Web 搜索
  - **Ambiguous**：质量不确定 → 组合两者
- ✅ **知识精炼**：Decompose-then-recompose 算法过滤无关信息

---

## ✨ 功能特点

### 核心功能

- **📄 PDF 文档管理**
  - 支持上传单个/批量 PDF 文档
  - 自动提取文本并分页处理
  - 支持中文和多语言文档

- **🔍 智能检索**
  - 向量语义检索（SentenceTransformer）
  - 关键词增强检索
  - 文档 ID 过滤

- **🧠 CRAG 评估**
  - 自动评估检索质量
  - 三种动作路由：Correct / Incorrect / Ambiguous
  - Decompose-then-recompose 知识精炼

- **🌐 Web 搜索增强**（可选）
  - 检索质量低时自动触发
  - 支持 Google / Bing / DuckDuckGo
  - 优先权威来源（Wikipedia、.edu、.gov）

### 高级特性

- **⚡ 性能优化**
  - 快速路径：高质量检索自动跳过昂贵的 CRAG 评估
  - 可配置：强制完整 CRAG 用于性能评估

- **🎯 精确控制**
  - 环境变量配置所有参数
  - 文档 ID 过滤
  - Top-K 结果数量控制

- **📊 详细日志**
  - 检索耗时统计
  - CRAG 评估过程
  - Action 判定结果

---

## 🏗️ 项目架构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         用户界面                              │
│                    (AgentOS / API)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agno Agent                               │
│  - 对话管理                                                   │
│  - 工具调用 (query_documents, upload_pdf, etc.)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG Tools Layer                           │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ upload_pdf   │  │query_documents│  │list_documents│      │
│  └──────────────┘  └───────┬───────┘  └──────────────┘      │
│                            │                                  │
│                            ▼                                  │
│              ┌──────────────────────────┐                    │
│              │  _query_documents_impl   │                    │
│              │  (检索 + CRAG 评估)      │                    │
│              └──────────┬───────────────┘                    │
└──────────────────────────┼──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ Knowledge   │   │  CRAG Core   │   │Vector Store │
│   Base      │   │              │   │  (LanceDB)  │
│ (Agno)      │   │ - Evaluator  │   │             │
│             │   │ - Router     │   │ Embedder:   │
│ - Contents  │   │ - Refiner    │   │ Sentence    │
│   DB        │   │ - Web Search │   │ Transformer │
│ - Vector DB │   │              │   │             │
└─────────────┘   └──────────────┘   └─────────────┘
```

### 目录结构

```
Agno-RAG/
├── agno_agent.py              # 主程序：Agent 和 AgentOS 配置
├── rag_tools.py               # RAG 工具：PDF 上传、文档查询
├── crag_core.py               # CRAG 核心组件
│   ├── SemanticRetrievalEvaluator  # T5 语义评估器
│   ├── CompleteActionRouter        # 动作路由器
│   ├── WebSearchAugmenter          # Web 搜索
│   └── EnhancedKnowledgeRefiner    # 知识精炼
├── crag_layer.py              # CRAG 基础算法
│   ├── crag_evaluate_and_route     # 评估和路由
│   ├── _split_into_strips          # 分解算法
│   └── create_t5_scorer            # T5 评分器创建
├── document_processor.py      # PDF 文档处理
├── persistent_vector_store.py # 持久化向量存储
├── upload_documents.py        # 批量上传工具
├── env.example                # 环境变量配置示例
└── pdf/                       # PDF 文档目录
```

---

## 🎓 CRAG 原理详解

### 传统 RAG vs CRAG

#### 传统 RAG 流程

```
查询 → 向量检索 → 返回 Top-K 文档 → LLM 生成答案
```

**问题**：
- 如果检索错误，LLM 会基于错误信息生成答案
- 无法判断检索质量
- 文档中包含大量无关信息

#### CRAG 流程

```
查询 → 向量检索 → CRAG 评估 → 动作路由 → 知识精炼 → LLM 生成答案
                      │
                      ├─ Correct → 精炼内部知识
                      ├─ Incorrect → Web 搜索
                      └─ Ambiguous → 组合两者
```

### CRAG 核心组件

#### 1. 检索评估器 (Retrieval Evaluator)

**作用**：评估每个检索文档与查询的相关性

**实现**：
- **轻量级**：使用 T5-large（0.77B 参数）或词法评分
- **评分**：为每个文档打分（-1.0 到 1.0）
- **快速**：支持批量评估

**判定逻辑**（符合论文）：
```python
if max(scores) >= upper_threshold (0.6):
    action = "correct"     # 至少一个文档高质量
elif max(scores) < lower_threshold (0.2):
    action = "incorrect"   # 所有文档低质量
else:
    action = "ambiguous"   # 质量不确定
```

#### 2. 知识精炼器 (Knowledge Refiner)

**作用**：过滤文档中的无关信息

**Decompose-then-recompose 算法**：

1. **Decompose（分解）**：
   ```
   长文档 → 按句子/固定长度分割 → 小片段
   ```

2. **Score（评分）**：
   ```
   为每个片段评分（相关性）
   ```

3. **Filter（过滤）**：
   ```
   保留高分片段，丢弃低分片段
   ```

4. **Recompose（重组）**：
   ```
   按顺序重新组合 → 精炼的知识
   ```

**三种分解模式**：
- **excerption**（推荐）：按句子分组（3 句/片段）
- **fixed_num**：固定长度（50 词/片段）
- **selection**：不分解，整体评分

#### 3. 动作路由器 (Action Router)

**三种动作策略**：

| Action | 条件 | 策略 | 适用场景 |
|--------|------|------|----------|
| **Correct** | 至少一个文档 ≥ 0.6 | 使用知识精炼 | 检索成功，文档相关 |
| **Incorrect** | 所有文档 < 0.2 | 使用 Web 搜索 | 检索失败，文档无关 |
| **Ambiguous** | 0.2 ≤ max < 0.6 | 组合两者 | 不确定，需要补充 |

**示例**：

```python
# 查询："GB146 标准中关于客车的安全要求是什么？"
# 文档分数：[0.85, 0.72, 0.35, 0.20, 0.15]

max_score = 0.85  # >= 0.6
→ Action: Correct
→ 策略: 精炼这 5 个文档，提取关键信息片段
```

#### 4. Web 搜索增强 (Web Search Augmenter)

**作用**：当内部检索失败时，从互联网获取知识

**流程**：
1. **查询重写**：将问题改写为搜索关键词
2. **Web 搜索**：使用 Google/Bing/DuckDuckGo
3. **优先过滤**：优先 Wikipedia、.edu、.gov
4. **内容提取**：爬取网页并提取文本
5. **知识精炼**：应用相同的精炼算法

---

## 🚀 快速开始

### 前置要求

- **Python 3.10+**
- **操作系统**：Windows / Linux / macOS
- **内存**：建议 4GB+
- **API Key**：SiliconFlow API Key（用于 LLM）

### 安装步骤

#### 1. 克隆项目

```bash
git clone <your-repo-url>
cd Agno-RAG
```

#### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 uv（更快）
pip install uv
uv pip install -r requirements.txt
```

**主要依赖**：
```
agno>=1.0.0              # Agno 框架
lancedb>=0.3.0           # 向量数据库
sentence-transformers    # 嵌入模型
transformers            # T5 评估器（可选）
torch                   # PyTorch
pypdf                   # PDF 处理
fastapi                 # Web API
uvicorn                 # ASGI 服务器
```

#### 4. 配置环境变量

创建 `.env` 文件：

```bash
# 复制示例文件
cp env.example .env
```

编辑 `.env`，添加必要配置：

```bash
# === 必需配置 ===

# LLM API Key（必需）
SILICONFLOW_API_KEY=your_api_key_here

# === CRAG 配置（可选，有默认值）===

# 启用完整 CRAG
USE_COMPLETE_CRAG=true

# 禁用快速路径（性能评估时使用）
DISABLE_FAST_PATH=false

# CRAG 参数
CRAG_TOP_K=5
CRAG_UPPER_THRESHOLD=0.6
CRAG_LOWER_THRESHOLD=0.2
CRAG_DECOMPOSE_MODE=excerption

# === 可选配置 ===

# Web 搜索（可选）
# GOOGLE_SEARCH_API_KEY=your_key
# GOOGLE_SEARCH_ENGINE_ID=your_id

# HuggingFace 镜像（国内加速）
# HF_ENDPOINT=https://hf-mirror.com
```

#### 5. 启动服务

```bash
python agno_agent.py
```

**成功启动后**，您会看到：

```
Agno Agent 已启动，日志系统已配置
============================================================
知识库已添加到 Agent: RAG Knowledge Base
  自动搜索: 已禁用（使用 query_documents 工具 + CRAG）
  ContentsDB: 已配置（AgentOS Knowledge 页面可用）

AgentOS 状态:
  - Agent 名称: Agno Agent
  - 工具数量: 6
  - 知识库: 已配置

访问地址: http://127.0.0.1:7777
API 文档: http://127.0.0.1:7777/docs
============================================================
```

#### 6. 访问界面

打开浏览器访问：
- **AgentOS 界面**：http://127.0.0.1:7777
- **API 文档**：http://127.0.0.1:7777/docs

---

## 📚 使用指南

### 基础使用

#### 1. 上传文档

**方法 A：通过 AgentOS 界面**

在对话框中输入：
```
上传 PDF 文档 C:\path\to\document.pdf
```

**方法 B：使用 upload_documents.py 脚本**

```bash
# 上传单个文件
python upload_documents.py --file path/to/document.pdf

# 上传整个目录
python upload_documents.py --dir path/to/pdf_folder

# 递归上传（包括子目录）
python upload_documents.py --dir path/to/pdf_folder --recursive
```

**方法 C：通过 API**

```bash
curl -X POST "http://127.0.0.1:7777/api/upload-pdf" \
  -F "file=@document.pdf"
```

#### 2. 查询文档

在 AgentOS 对话框中提问：

```
GB146 标准中关于客车的安全要求是什么？
```

**Agent 会自动**：
1. 调用 `query_documents` 工具
2. 执行向量检索
3. 运行 CRAG 评估
4. 精炼知识片段
5. 生成答案

#### 3. 查看上传的文档

```
列出所有文档
```

或

```
list_documents
```

#### 4. 删除文档

```
删除文档 GB146
```

### 高级功能

#### 指定文档 ID 查询

```
在 GB146 文档中查询客车安全要求
```

系统会自动识别文档 ID 并过滤结果。

#### 清空知识库

```
清空知识库
```

⚠️ 警告：此操作不可撤销！

---

## ⚙️ 配置说明

### CRAG 模式选择

#### 模式 1：优化 CRAG（推荐用于生产）

```bash
USE_COMPLETE_CRAG=true
DISABLE_FAST_PATH=false  # 启用快速路径优化
```

**特点**：
- ✅ 高质量检索走快速路径（跳过昂贵的评估）
- ✅ 低质量检索使用完整 CRAG
- ✅ 平衡速度和质量

**快速路径触发条件**：
1. 检索分数很高（max ≥ 0.5 或 avg ≥ 0.4）
2. 找到匹配的文档 ID
3. 结果包含查询关键词

#### 模式 2：完整 CRAG（用于性能评估）

```bash
USE_COMPLETE_CRAG=true
DISABLE_FAST_PATH=true   # 禁用快速路径
```

**特点**：
- ✅ 所有查询使用完整 CRAG
- ✅ 可以评估 CRAG 性能
- ⚠️ 速度稍慢

**日志输出**：
```
[性能评估模式] 快速路径已禁用，强制使用完整 CRAG
CRAG评估模式: 完整CRAG
  动作路由开始: 10 个文档
  文档分数范围: [0.1234, 0.8765]
  [完成] 决策动作: correct
```

#### 模式 3：基础检索（最快）

```bash
USE_COMPLETE_CRAG=false
```

**特点**：
- ✅ 仅使用向量检索
- ✅ 速度最快
- ⚠️ 没有 CRAG 优化

### 环境变量完整列表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| **LLM 配置** | | |
| `SILICONFLOW_API_KEY` | 无 | **必需**：SiliconFlow API Key |
| **CRAG 核心** | | |
| `USE_COMPLETE_CRAG` | `true` | 启用完整 CRAG |
| `DISABLE_FAST_PATH` | `false` | 禁用快速路径优化 |
| `VERBOSE_CRAG` | `false` | 详细日志 |
| **CRAG 参数** | | |
| `CRAG_TOP_K` | `5` | 返回的知识片段数 |
| `CRAG_SIMILARITY_THRESHOLD` | `0.15` | 向量检索相似度阈值 |
| `CRAG_UPPER_THRESHOLD` | `0.6` | Correct action 阈值 |
| `CRAG_LOWER_THRESHOLD` | `0.2` | Incorrect action 阈值 |
| `CRAG_DECOMPOSE_MODE` | `excerption` | 分解模式 |
| **Web 搜索** | | |
| `GOOGLE_SEARCH_API_KEY` | 无 | Google Search API Key |
| `GOOGLE_SEARCH_ENGINE_ID` | 无 | Google 搜索引擎 ID |
| **其他** | | |
| `HF_ENDPOINT` | 无 | HuggingFace 镜像地址 |

### 性能调优建议

#### 提高速度

1. **启用快速路径**：
   ```bash
   DISABLE_FAST_PATH=false
   ```

2. **减少 Top-K**：
   ```bash
   CRAG_TOP_K=3
   ```

3. **使用 selection 模式**（不分解）：
   ```bash
   CRAG_DECOMPOSE_MODE=selection
   ```

#### 提高质量

1. **使用完整 CRAG**：
   ```bash
   USE_COMPLETE_CRAG=true
   DISABLE_FAST_PATH=true
   ```

2. **增加 Top-K**：
   ```bash
   CRAG_TOP_K=10
   ```

3. **使用 excerption 模式**（推荐）：
   ```bash
   CRAG_DECOMPOSE_MODE=excerption
   ```

4. **配置 Web 搜索**（提供外部知识）：
   ```bash
   GOOGLE_SEARCH_API_KEY=your_key
   GOOGLE_SEARCH_ENGINE_ID=your_id
   ```

---

## 🔍 日志解读

### 完整查询日志示例

```
============================================================
开始查询文档
============================================================
查询: GB146 标准中关于客车的安全要求是什么？
参数: top_k=5, similarity_threshold=0.15, decompose_mode=excerption
------------------------------------------------------------

[检索阶段 1/5] 向量搜索...
检索到文档数: 10
文档ID分布: {'GB146': 8, 'GB10494': 2}

提取候选段落: 0.123秒
候选段落数: 10

[性能评估模式] 快速路径已禁用，强制使用完整 CRAG

CRAG评估模式: 完整CRAG

  动作路由开始: 10 个文档
  质量评估: 0.234秒
  文档分数范围: [0.1523, 0.8234], 平均: 0.4321
  [完成] 决策动作: correct (阈值: upper=0.6, lower=0.2)
  最高文档分数: 0.8234
  动作路由总耗时: 0.456秒

CRAG评估总耗时: 0.456秒

结果格式化: 0.012秒

============================================================
[完成] 检索完成 - 总耗时: 0.89秒
   检索: 0.40秒 (44.9%)
   CRAG评估: 0.46秒 (51.7%)
   其他: 0.03秒
============================================================
```

### 日志分析

- **检索阶段**：向量搜索，找到 10 个相关文档
- **CRAG 评估**：
  - 文档分数范围：0.15 - 0.82
  - 最高分 0.82 > 0.6 → **Correct action**
  - 策略：使用知识精炼
- **性能**：总耗时 0.89 秒，其中 CRAG 评估占 51.7%

---

## 🤝 常见问题

### Q1: 如何加速首次模型下载？

**A**: 使用 HuggingFace 镜像

```bash
# 在 .env 中添加
HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 为什么查询很慢？

**A**: 可能原因：

1. **完整 CRAG 模式**：尝试启用快速路径
   ```bash
   DISABLE_FAST_PATH=false
   ```

2. **T5 evaluator 在 CPU 上运行**：考虑使用 GPU 或禁用 T5

3. **Top-K 太大**：减少返回数量
   ```bash
   CRAG_TOP_K=3
   ```

### Q3: 如何评估 CRAG 性能？

**A**: 启用完整 CRAG 模式并观察日志

```bash
# .env
DISABLE_FAST_PATH=true
VERBOSE_CRAG=true
```

观察：
- Action 分布（Correct/Incorrect/Ambiguous）
- 检索质量（文档分数）
- 耗时分布

### Q4: PDF 上传失败怎么办？

**A**: 检查：

1. PDF 格式是否正确
2. PDF 是否包含可提取的文本（不是纯图片）
3. 文件路径是否正确

### Q5: 如何使用自己的 LLM？

**A**: 修改 `agno_agent.py` 中的模型配置：

```python
# 使用 OpenAI
from agno.models.openai import OpenAIChat
llm_model = OpenAIChat(id="gpt-4", api_key="your_key")

# 使用本地 Ollama
from agno.models.ollama import Ollama
llm_model = Ollama(id="llama3")
```

---

## 📄 参考文献

1. **CRAG 论文**:
   - Yan, S. Q., et al. (2024). "Corrective Retrieval Augmented Generation". *arXiv:2401.15884v3*.
   - 链接: https://arxiv.org/abs/2401.15884

2. **Agno Framework**:
   - 文档: https://docs.agno.com

3. **LanceDB**:
   - 文档: https://lancedb.github.io/lancedb/

---

## 📝 License

MIT License

---

## 🙏 致谢

- **CRAG 论文作者**：Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
- **Agno Framework**：提供强大的 Agent 框架
- **SentenceTransformers**：提供多语言嵌入模型
- **LanceDB**：提供高性能向量数据库

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star！**

</div>
