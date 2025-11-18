# AGNO-CRAG - CRAG 增强的检索增强生成系统

一个基于 Agno 框架的智能文档问答系统，集成了 CRAG (Corrective Retrieval Augmented Generation) 技术，支持 PDF 文档上传、智能检索和知识问答。

## 📋 目录

- [功能特性](#功能特性)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [AgentOS 连接](#agnoos-连接)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [技术架构](#技术架构)

## ✨ 功能特性

### 核心功能

1. **PDF 文档管理**
   - 单文件上传：支持上传单个 PDF 文档
   - 批量上传：支持上传整个目录下的所有 PDF 文件
   - 文档列表：查看所有已上传的文档
   - 文档删除：删除指定文档
   - 清空知识库：一键清空所有文档（谨慎使用）

2. **智能检索 (CRAG)**
   - 语义检索：基于向量相似度的智能文档检索
   - CRAG 评估：自动评估检索结果的质量
   - 文档分解：将长文档分解为更小的片段进行精确匹配
   - 智能路由：根据检索质量自动选择最佳知识源

3. **AgentOS 集成**
   - Web 界面：通过 AgentOS 进行可视化管理
   - 知识库管理：在 AgentOS 中直接上传和管理文档
   - 实时对话：与智能体进行实时问答交互

## 🖥️ 系统要求

### 必需环境

- **Python**: 3.10 或更高版本
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议 8GB 以上（使用 T5 评估器需要更多）
- **存储**: 至少 2GB 可用空间

### 推荐配置

- **GPU**: 可选，但使用 T5 语义评估器时强烈推荐（NVIDIA GPU，支持 CUDA）
- **内存**: 16GB 或更多（使用完整 CRAG 时）

## 🚀 快速开始

### 步骤 1: 克隆项目

```bash
git clone <your-repo-url>
cd Agno-RAG
```

### 步骤 2: 安装依赖管理工具

本项目使用 `uv` 作为依赖管理工具。如果还没有安装，请先安装：

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**或者使用 pip:**
```bash
pip install uv
```

### 步骤 3: 安装项目依赖

```bash
uv sync
```

这将自动创建虚拟环境并安装所有必需的依赖包。

### 步骤 4: 配置环境变量

1. **复制配置模板**：
   ```bash
   cp .env.example .env
   ```

2. **编辑 `.env` 文件**，填写您的 API 密钥：
   ```env
   # 必需：DeepSeek API 密钥（通过 SiliconFlow）
   DEEPSEEK_API_KEY=your_api_key_here
   ```

   > 💡 **获取 API 密钥**：
   > - 访问 [SiliconFlow](https://siliconflow.cn/) 注册账号
   > - 在控制台创建 API 密钥
   > - 将密钥复制到 `.env` 文件中

### 步骤 5: 启动服务

```bash
uv run uvicorn agno_agent:app --host 0.0.0.0 --port 7777 --reload
```

如果一切正常，您应该看到类似以下的输出：

```
============================================================
Agno Agent 已启动，日志系统已配置
============================================================
检索日志将输出到此控制台
============================================================

AgentOS 状态:
  - Agent 名称: Agno Agent
  - 数据库: SQLite (C:\...\tmp\data.db)
  - 工具数量: 6
  - 知识库: 已配置
  - 知识库名称: RAG Knowledge Base
  - 知识库内容: 0 项
  - 知识库状态: 空（启动后将添加占位符内容以确保 AgentOS 识别）
  - ContentsDB: 已配置（AgentOS Knowledge 页面应可用）
  - FastAPI 应用: 已创建

访问地址: http://127.0.0.1:7777
API 文档: http://127.0.0.1:7777/docs
============================================================
```

### 步骤 6: 访问服务

- **AgentOS Web 界面**: http://127.0.0.1:7777
- **API 文档**: http://127.0.0.1:7777/docs
- **AgentOS 云端连接**: https://os.agno.com（需要配置连接）

## ⚙️ 详细配置

### 环境变量配置

所有配置都在 `.env` 文件中。以下是所有可配置项：

#### 必需配置

```env
# DeepSeek 模型配置（通过 SiliconFlow）
DEEPSEEK_MODEL_ID=deepseek-ai/DeepSeek-V3.1-Terminus
DEEPSEEK_API_KEY=your_api_key_here  # ⚠️ 必需！
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
```

#### CRAG 检索策略配置

```env
# 是否使用完整 CRAG（包含 T5 语义评估器）
# true: 使用完整 CRAG（推荐，但需要 T5 模型）
# false: 使用基础 CRAG（仅词法评估，更快但准确度较低）
USE_COMPLETE_CRAG=true

# 文档分解模式
# - 'fixed_num': 固定数量分割
# - 'excerption': 提取关键摘要（推荐）
# - 'selection': 选择 top-k 片段（更快但准确度较低）
CRAG_DECOMPOSE_MODE=excerption

# 返回的 top-k 段落数量
CRAG_TOP_K=5

# 检索的最小相似度分数（0.0-1.0）
CRAG_SIMILARITY_THRESHOLD=0.15

# 'correct' 动作的阈值（分数高于此值视为正确）
CRAG_UPPER_THRESHOLD=0.6

# 'ambiguous' 动作的阈值（分数低于此值视为不正确）
CRAG_LOWER_THRESHOLD=0.2
```

#### 可选配置

```env
# Google 搜索 API（用于 CRAG 外部知识检索）
# GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here

# T5 评估器配置
# T5_EVALUATOR_PATH=finetuned_t5_evaluator
# T5_BATCH_SIZE=4
```

### 配置说明

#### CRAG 模式选择

- **完整 CRAG (`USE_COMPLETE_CRAG=true`)**
  - ✅ 优点：准确度最高，使用 T5 模型进行语义评估
  - ❌ 缺点：速度较慢，需要 T5 模型文件
  - 📦 需要：`finetuned_t5_evaluator` 目录中的 T5 模型

- **基础 CRAG (`USE_COMPLETE_CRAG=false`)**
  - ✅ 优点：速度快，无需额外模型
  - ❌ 缺点：准确度较低，仅使用词法匹配

#### 分解模式选择

- **`excerption`**（推荐）：提取关键摘要，平衡速度和准确度
- **`selection`**：快速模式，适合大量文档
- **`fixed_num`**：固定数量分割，适合结构化文档

## 🔗 AgentOS 连接

### 本地连接

服务启动后，AgentOS 会自动在本地运行，访问地址：
- http://127.0.0.1:7777

### 云端连接（可选）

如果您想通过 Agno 官方的 AgentOS 云端界面连接：

1. **确保服务正在运行**（步骤 5）

2. **访问 AgentOS 云端**：
   - 打开 https://os.agno.com
   - 登录您的 Agno 账号

3. **添加本地 Agent**：
   - 在 AgentOS 界面中，点击"添加 Agent"或"连接本地 Agent"
   - 输入本地服务地址：`http://your-ip:7777`
   - 如果服务在同一台机器上，使用：`http://127.0.0.1:7777`

4. **验证连接**：
   - 连接成功后，您应该能看到 "Agno RAG Agent"
   - 在 Knowledge 部分可以看到知识库

### 连接问题排查

如果无法连接到 AgentOS：

1. **检查服务是否运行**：
   ```bash
   # 检查端口是否被占用
   netstat -ano | findstr :7777  # Windows
   lsof -i :7777                 # Linux/macOS
   ```

2. **检查防火墙设置**：
   - 确保防火墙允许 7777 端口的连接

3. **检查 CORS 配置**：
   - 确保 `.env` 中没有修改 CORS 相关配置
   - 检查 `agno_agent.py` 中的 CORS 中间件配置

4. **查看日志**：
   - 检查控制台输出的错误信息
   - 查看是否有 "CORS 中间件已配置" 的日志

## 📖 使用指南

### 方式 1: 通过 AgentOS Web 界面

1. **访问界面**：打开 http://127.0.0.1:7777

2. **上传文档**：
   - 在 Knowledge 部分点击"上传"
   - 选择 PDF 文件或拖拽文件到上传区域
   - 等待处理完成

3. **提问**：
   - 在对话界面输入您的问题
   - 智能体会自动使用 `query_documents` 工具检索相关文档
   - 基于检索结果生成答案

### 方式 2: 通过 API 调用

#### 上传文档

```bash
# 单文件上传
curl -X POST "http://127.0.0.1:7777/api/upload-pdf" \
  -F "file=@/path/to/your/document.pdf"

# 或使用 Python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:7777/api/upload-pdf",
        files={"file": f}
    )
print(response.json())
```

#### 查询文档

```python
from rag_tools import query_documents

# 简单查询
result = query_documents(
    query="GB 12352-2018 中关于运行速度的规定是什么？"
)

# 带参数的查询
result = query_documents(
    query="客运架空索道安全规范",
    top_k=10,
    similarity_threshold=0.2,
    decompose_mode="excerption",
    doc_id_filter="GB12352"
)

print(result["context"])  # 查看检索到的上下文
```

### 方式 3: 使用管理脚本

项目包含一个管理脚本 `upload_documents.py`：

```bash
# 上传单个文件
uv run python upload_documents.py upload "path/to/document.pdf"

# 批量上传目录
uv run python upload_documents.py upload-dir "path/to/pdf/directory"

# 列出所有文档
uv run python upload_documents.py list

# 清空知识库
uv run python upload_documents.py clear
```

### 可用工具

智能体可以使用以下工具：

1. **`upload_pdf_document`**: 上传单个 PDF 文档
2. **`upload_pdf_directory`**: 批量上传目录中的 PDF 文件
3. **`query_documents`**: 查询知识库（使用 CRAG 检索）
4. **`list_documents`**: 列出所有已上传的文档
5. **`delete_document`**: 删除指定文档
6. **`clear_knowledge_base`**: 清空整个知识库

## 📁 项目结构

```
Agno-RAG/
├── agno_agent.py              # 主应用入口，AgentOS 配置
├── rag_tools.py               # RAG 工具定义（上传、查询等）
├── crag_core.py               # CRAG 核心组件（语义评估器、动作路由等）
├── crag_layer.py              # CRAG 基础层（词法评估、文档分解）
├── document_processor.py      # PDF 文档处理
├── persistent_vector_store.py # 向量存储（向后兼容）
├── knowledge_agent.py         # 知识库测试脚本
├── upload_documents.py        # 管理脚本
├── pyproject.toml             # 项目配置和依赖
├── .env.example               # 环境变量模板
├── .env                       # 实际配置文件（不提交到 Git）
├── .gitignore                 # Git 忽略文件
├── README.md                  # 本文件
└── tmp/                       # 临时文件目录
    ├── data.db                # Agent 会话数据库
    ├── knowledge_contents.db  # 知识库内容数据库
    └── lancedb/               # LanceDB 向量数据库
```

## ❓ 常见问题

### Q1: 启动时提示 "DEEPSEEK_API_KEY is required"

**原因**：未配置 API 密钥

**解决**：
1. 确保已创建 `.env` 文件（从 `.env.example` 复制）
2. 在 `.env` 文件中填写 `DEEPSEEK_API_KEY=your_actual_key`
3. 重启服务

### Q2: 无法连接到 AgentOS

**原因**：可能是端口被占用或 CORS 配置问题

**解决**：
1. 检查 7777 端口是否被占用
2. 尝试更改端口：`uvicorn agno_agent:app --port 8000`
3. 检查防火墙设置
4. 查看控制台日志中的错误信息

### Q3: 上传 PDF 后无法检索到内容

**原因**：可能是文档处理失败或检索参数设置不当

**解决**：
1. 检查文档是否成功上传（使用 `list_documents` 工具）
2. 降低 `CRAG_SIMILARITY_THRESHOLD` 值（如改为 0.1）
3. 检查日志中的错误信息
4. 确保 PDF 文件不是扫描件（需要可提取的文本）

### Q4: CRAG 评估速度很慢

**原因**：使用完整 CRAG 且没有 GPU 加速

**解决**：
1. 设置 `USE_COMPLETE_CRAG=false` 使用基础 CRAG
2. 如果有 GPU，确保 PyTorch 正确识别 GPU
3. 调整 `CRAG_DECOMPOSE_MODE=selection` 使用快速模式
4. 减少 `CRAG_TOP_K` 值

### Q5: 提示 "No module named 'xxx'"

**原因**：依赖未正确安装

**解决**：
```bash
# 重新安装依赖
uv sync

# 或手动安装缺失的包
uv pip install <package-name>
```

### Q6: 知识库显示 "No databases found"

**原因**：ContentsDB 未正确初始化

**解决**：
1. 检查 `tmp/knowledge_contents.db` 文件是否存在
2. 重启服务，查看启动日志
3. 确保知识库初始化成功（查看日志中的 "ContentsDB 已配置" 消息）

## 🏗️ 技术架构

### 核心组件

1. **Agno Framework**: 提供 Agent 和 AgentOS 基础框架
2. **LanceDB**: 向量数据库，用于存储文档嵌入
3. **Sentence Transformers**: 本地嵌入模型（支持中英文）
4. **CRAG**: 纠正性检索增强生成算法
   - 语义评估器（T5 模型，可选）
   - 动作路由系统
   - 文档分解与重组

### 工作流程

```
用户查询
    ↓
向量检索（LanceDB）
    ↓
CRAG 评估
    ├─ 快速路径（高质量结果）
    │   └─ 词法评估 → 返回结果
    └─ 完整路径（需要评估）
        ├─ T5 语义评估（可选）
        ├─ 动作路由（correct/incorrect/ambiguous）
        └─ 文档分解与重组
    ↓
生成答案
```

### 数据流

1. **文档上传**：
   - PDF → 文本提取 → 分块 → 嵌入 → 存储到 LanceDB

2. **查询处理**：
   - 查询 → 嵌入 → 向量搜索 → CRAG 评估 → 上下文提取 → LLM 生成

---

**祝您使用愉快！** 🎉
