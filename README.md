# Agno-RAG: CRAG增强的智能文档问答系统

> 一个基于Agno框架和CRAG（Corrective Retrieval Augmented Generation）技术的高性能文档问答系统

## 📖 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [CRAG策略详解](#crag策略详解)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [配置说明](#配置说明)
- [AgentOS连接](#agentos连接)
- [策略配置](#策略配置)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 项目简介

Agno-RAG 是一个智能文档问答系统，可以让AI理解和回答您上传的PDF文档中的内容。与传统的RAG系统不同，本系统采用了**CRAG（校正检索增强生成）**技术，能够智能评估检索结果的质量，并自动采取最优策略，大幅提升回答的准确性。

### 适用场景

- 📚 技术文档查询（标准规范、技术手册）
- 📄 企业知识库管理
- 🔍 合同文件分析
- 📖 学术论文阅读助手

---

## 核心功能

### 1. 智能文档管理

- ✅ **PDF上传与解析**：支持批量上传PDF文档
- ✅ **智能分块**：自动将文档切分为语义连贯的片段
- ✅ **向量存储**：使用LanceDB进行高效的向量检索
- ✅ **文档追踪**：支持按文档ID过滤查询

### 2. CRAG增强检索

本系统的核心创新是**CRAG（Corrective RAG）**，它包含三个关键组件：

#### 🎯 语义检索评估器（T5-based）
- 使用微调的T5模型评估检索文档与查询的语义相关性
- 返回连续分数（-1到1），比简单的关键词匹配更准确
- **支持GPU加速**，评估速度提升10-40倍

#### 🔀 三动作路由机制
根据文档质量自动选择最优策略：

| 动作 | 触发条件 | 处理方式 |
|------|---------|---------|
| **Correct** | 高质量文档（分数>0.6） | 直接使用检索结果 |
| **Incorrect** | 低质量文档（分数<0.2） | 触发外部知识搜索 |
| **Ambiguous** | 中等质量文档（0.2≤分数≤0.6） | 知识精炼与重构 |

#### 🔧 知识精炼器
- **分解-重构**策略：将长文档拆分为知识片段
- **语义过滤**：去除与查询无关的内容
- **信息重组**：重新组织知识以提供更清晰的答案

### 3. 查询增强策略

- **查询扩展**：自动生成相似查询以提高召回率
- **多策略检索**：
  - 原始查询检索
  - 文档ID过滤
  - 混合检索
- **结果去重与排序**：智能合并和排序检索结果

### 4. Web界面（AgentOS集成）

- 🌐 现代化的Web界面
- 💬 对话式交互
- 📁 知识库可视化管理
- 📊 查询历史追踪

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户界面                              │
│                  (AgentOS Web Interface)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agno Agent                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RAG Tools (rag_tools.py)               │   │
│  │  • upload_pdf_document                               │   │
│  │  • query_documents (CRAG-enabled)                   │   │
│  │  • list_documents                                    │   │
│  │  • delete_document                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Vector Store │ │  CRAG Layer  │ │  LLM Model   │
│  (LanceDB)   │ │              │ │ (DeepSeek)   │
│              │ │ ┌──────────┐ │ │              │
│ • Embeddings │ │ │T5 Eval   │ │ │              │
│ • Documents  │ │ │Router    │ │ │              │
│ • Passages   │ │ │Refiner   │ │ │              │
└──────────────┘ └─┴──────────┘─┘ └──────────────┘
```

### 数据流程

1. **文档上传** → PDF解析 → 文本提取 → 分块 → 向量化 → 存储
2. **用户查询** → 向量检索 → CRAG评估 → 动作路由 → 知识精炼 → LLM生成答案

---

## CRAG策略详解

### 为什么需要CRAG？

传统RAG系统的问题：
- ❌ 不区分检索结果的质量
- ❌ 低质量文档会误导AI
- ❌ 无法处理知识库缺失的情况
- ❌ 冗余信息干扰答案生成

CRAG的解决方案：
- ✅ **智能评估**：自动判断检索结果是否可靠
- ✅ **动态路由**：根据质量选择不同处理策略
- ✅ **知识精炼**：提取关键信息，过滤噪音
- ✅ **外部补充**：质量不佳时触发web搜索

### CRAG工作流程

```
查询 → 向量检索
         │
         ▼
    CRAG评估
         │
    ┌────┴────┐
    ▼         ▼
语义评估    快速路径
(T5模型)   (词法评分)
    │         │
    └────┬────┘
         ▼
    质量评分 (confidence score)
         │
    ┌────┼────┐
    ▼    ▼    ▼
 Correct Ambiguous Incorrect
    │    │         │
    │    ▼         ▼
    │  知识精炼  Web搜索
    │    │         │
    └────┴────┬────┘
              ▼
          生成答案
```

### 三种评估模式

#### 1. 快速路径（Fast Path）
- **触发条件**：检索结果分数很高（>0.95）且文档数量适中
- **特点**：使用词法评分，速度快
- **适用场景**：高置信度查询，追求响应速度

#### 2. 完整CRAG（Full CRAG）
- **触发条件**：默认模式或检索质量不确定
- **特点**：使用T5语义评估，准确度高
- **适用场景**：需要高准确度的查询

#### 3. 性能评估模式（Performance Mode）
- **触发条件**：设置 `DISABLE_FAST_PATH=true`
- **特点**：强制使用完整CRAG，便于性能测试
- **适用场景**：开发调试、性能优化

---

## 快速开始

### 环境要求

#### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4核心 | 8核心+ |
| 内存 | 8GB | 16GB+ |
| 硬盘 | 20GB可用空间 | 50GB+ SSD |
| GPU | 无（使用CPU） | NVIDIA GPU 6GB+ 显存 |

**GPU支持**：
- ✅ 推荐：NVIDIA GPU（RTX 3060及以上）
- ⚡ 性能提升：使用GPU可将T5评估速度提升10-40倍
- 📝 支持的CUDA版本：11.8 或 12.1

#### 软件要求

- **操作系统**：Windows 10/11、Linux、macOS
- **Python**：3.10 或更高版本
- **包管理器**：uv（推荐）或 pip

---

### 安装步骤

#### 步骤 1: 克隆项目

```bash
# 克隆仓库
git clone https://github.com/your-username/Agno-RAG.git
cd Agno-RAG
```

#### 步骤 2: 安装uv（推荐的包管理器）

**Windows (PowerShell)**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 步骤 3: 安装依赖

**基础安装（CPU版本）**:
```bash
uv sync
```

**GPU版本安装**（推荐，性能更好）:

如果您有NVIDIA GPU，项目已配置使用GPU版本的PyTorch：

```bash
# 1. 首先确认GPU可用
nvidia-smi

# 2. 同步安装（已配置GPU索引）
uv sync

# 3. 验证GPU支持
uv run python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

**注意**：
- 项目的 `pyproject.toml` 已配置PyTorch GPU索引
- 无需手动指定索引URL
- 如果验证失败，查看[GPU配置故障排查](#gpu配置)

---

### 配置说明

#### 步骤 4: 配置环境变量

创建 `.env` 文件（从模板复制）：

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

然后编辑 `.env` 文件：

```bash
# ============================================================
# 核心配置
# ============================================================

# DeepSeek API配置（必需）
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
DEEPSEEK_MODEL_ID=deepseek-ai/DeepSeek-V3.1-Terminus

# ============================================================
# CRAG配置
# ============================================================

# 启用完整CRAG（推荐）
USE_COMPLETE_CRAG=true

# 启用T5语义评估器
ENABLE_T5_EVALUATOR=true

# T5模型路径
T5_EVALUATOR_PATH=finetuned_t5_evaluator

# T5批处理大小
# CPU: 4-8, GPU(6GB): 8-12, GPU(8GB): 12-16, GPU(12GB+): 16-24
T5_BATCH_SIZE=12

# 禁用快速路径（强制使用完整CRAG，便于性能测试）
DISABLE_FAST_PATH=false

# ============================================================
# Web搜索配置（可选）
# ============================================================

# 启用Web搜索增强
ENABLE_WEB_SEARCH=false

# SerpAPI Key（如果启用web搜索）
# SERPAPI_KEY=your-serpapi-key

# ============================================================
# 性能配置
# ============================================================

# 详细日志输出
VERBOSE_CRAG=false
```

#### 步骤 5: 获取API密钥

##### 1. DeepSeek API（必需）

本项目使用SiliconFlow提供的DeepSeek API：

1. 访问 [SiliconFlow](https://cloud.siliconflow.cn/)
2. 注册账号并登录
3. 进入「API密钥」页面
4. 创建新的API密钥
5. 复制密钥到 `.env` 文件的 `DEEPSEEK_API_KEY`

**费用说明**：
- 新用户通常有免费额度
- 按Token计费，成本较低
- 详细价格见官网

##### 2. SerpAPI（可选，用于Web搜索）

如果需要外部知识增强：

1. 访问 [SerpAPI](https://serpapi.com/)
2. 注册账号
3. 获取API密钥
4. 在 `.env` 中设置：
   ```bash
   ENABLE_WEB_SEARCH=true
   SERPAPI_KEY=your-serpapi-key
   ```

#### 步骤 6: 准备T5模型

本系统需要微调的T5评估器模型：

**选项 A：使用预训练模型**（推荐）

如果您有预训练的T5模型，将其放在项目根目录：

```
Agno-RAG/
├── finetuned_t5_evaluator/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── spiece.model
└── ...
```

**选项 B：禁用T5评估器**（降级模式）

如果暂时没有T5模型，可以使用词法评分：

在 `.env` 中设置：
```bash
ENABLE_T5_EVALUATOR=false
```

**注意**：禁用T5会降低评估准确度，建议仅用于测试。

---

## AgentOS连接

AgentOS 是Agno框架提供的现代化Web界面，让您可以通过浏览器与AI对话。

### 启动服务

```bash
# 使用uv运行（推荐）
uv run python agno_agent.py

# 或直接运行（如果已激活虚拟环境）
python agno_agent.py
```

### 访问界面

启动成功后，您会看到：

```
模型配置已加载: deepseek-ai/DeepSeek-V3.1-Terminus
SQLite 数据库已配置
...
知识库已创建（使用 LanceDB）
...

访问地址: http://127.0.0.1:7777
API 文档: http://127.0.0.1:7777/docs
```

#### 本地访问

在浏览器中打开：`http://127.0.0.1:7777`

您会看到Agno的本地调试界面。

#### 连接到AgentOS云端

1. **注册AgentOS账号**
   - 访问 [os.agno.com](https://os.agno.com)
   - 注册并登录

2. **添加Agent连接**
   - 点击 「Add Agent」或「添加智能体」
   - 选择 「Local Agent」
   - 输入连接信息：
     ```
     名称: Agno-RAG
     URL: http://127.0.0.1:7777
     ```
   - 点击「连接」

3. **开始使用**
   - 连接成功后，您可以：
     - 💬 与AI对话
     - 📁 在「Knowledge」标签页管理文档
     - 📊 查看对话历史
     - ⚙️ 调整Agent设置

### AgentOS功能说明

#### 1. 对话界面（Chat）

- 输入问题，AI会自动调用 `query_documents` 工具
- 支持多轮对话，AI会记住上下文
- 可以随时上传新文档

#### 2. 知识库管理（Knowledge）

- **查看文档**：显示所有已上传的文档
- **上传文档**：
  - 点击「Upload」按钮
  - 选择PDF文件（支持多选）
  - 系统自动解析并存储
- **删除文档**：点击文档旁的删除按钮

#### 3. 工具调用

AI会自动调用以下工具：

- `query_documents`：查询文档内容（自动使用CRAG）
- `upload_pdf_document`：上传单个PDF
- `list_documents`：列出所有文档
- `delete_document`：删除指定文档

您可以在对话中直接说：
- "上传这个PDF"
- "列出所有文档"
- "删除文档ID为xxx的文档"

---

## 策略配置

### CRAG核心参数

#### 1. 评估器配置

```bash
# 启用/禁用T5评估器
ENABLE_T5_EVALUATOR=true

# T5批处理大小（影响GPU利用率）
# 建议值：
#   CPU: 4-8
#   GPU 6GB: 8-12
#   GPU 8GB: 12-16
#   GPU 12GB+: 16-32
T5_BATCH_SIZE=12

# T5模型路径
T5_EVALUATOR_PATH=finetuned_t5_evaluator
```

#### 2. 动作路由阈值

在 `rag_tools.py` 的 `get_action_router()` 函数中配置：

```python
router = CompleteActionRouter(
    evaluator=evaluator,
    web_searcher=web_searcher,
    upper_threshold=0.6,  # Correct阈值（高于此值=高质量）
    lower_threshold=0.2,  # Incorrect阈值（低于此值=低质量）
)
```

**阈值调整建议**：

| 场景 | upper_threshold | lower_threshold | 说明 |
|------|----------------|----------------|------|
| 严格模式 | 0.7 | 0.3 | 更多ambiguous，更多精炼 |
| 平衡模式 | 0.6 | 0.2 | 默认，平衡准确度和性能 |
| 宽松模式 | 0.5 | 0.1 | 更多correct，更快响应 |

#### 3. 检索参数

在对话中查询时，系统默认使用：

```python
query_documents(
    query="您的问题",
    top_k=10,        # 检索文档数量
    threshold=0.15,  # 相似度阈值
    mode="excerption" # 知识精炼模式
)
```

**参数说明**：

- **top_k**（5-20）：
  - 越大：召回率更高，但速度更慢
  - 越小：速度更快，但可能遗漏相关文档
  - 推荐：10

- **threshold**（0.0-1.0）：
  - 越高：只返回高相似度文档，可能召回不足
  - 越低：返回更多文档，可能包含噪音
  - 推荐：0.15

- **mode**（excerption/original）：
  - `excerption`：启用知识精炼（推荐）
  - `original`：使用原始文档

#### 4. 快速路径配置

```bash
# 禁用快速路径（强制完整CRAG）
DISABLE_FAST_PATH=false

# 快速路径触发条件（在crag_layer.py中）
FAST_PATH_SCORE_THRESHOLD=0.95  # 检索分数阈值
FAST_PATH_MAX_DOCS=15           # 最大文档数
```

### Web搜索配置

```bash
# 启用Web搜索
ENABLE_WEB_SEARCH=true

# SerpAPI配置
SERPAPI_KEY=your-key

# Web搜索参数（在rag_tools.py中）
WEB_SEARCH_NUM_RESULTS=5  # 搜索结果数
```

### 日志和调试

```bash
# 启用详细日志
VERBOSE_CRAG=true

# 日志会显示：
# - 检索过程细节
# - CRAG评估分数
# - 动作路由决策
# - 知识精炼结果
# - 性能统计
```

---

## 性能优化

### GPU加速配置

#### 检查GPU支持

```bash
# 检查GPU
nvidia-smi

# 验证PyTorch GPU支持
uv run python -c "import torch; print(torch.cuda.is_available())"
```

#### GPU配置

项目已自动配置GPU支持。如果遇到问题：

1. **确认安装GPU版本torch**：
   ```bash
   uv pip list | grep torch
   # 应显示: torch 2.5.1+cu121
   ```

2. **如果显示CPU版本**：
   ```bash
   # 重新安装
   uv pip uninstall torch
   uv sync
   ```

3. **调整批处理大小**：
   ```bash
   # .env中，根据GPU显存调整
   T5_BATCH_SIZE=16  # 8GB显存
   ```

### 性能基准

| 配置 | 10文档评估耗时 | 吞吐量 | 提升 |
|------|--------------|--------|------|
| CPU (i7) | ~30-40秒 | ~3文档/秒 | 基准 |
| GPU (RTX 3060 6GB) | ~2-3秒 | ~30文档/秒 | 10倍 |
| GPU (RTX 3070 8GB) | ~1-2秒 | ~50文档/秒 | 15倍 |
| GPU (RTX 4070 12GB) | ~0.5-1秒 | ~100文档/秒 | 30倍 |

### 缓存优化

系统自动缓存：
- 文档向量
- T5评估器实例
- 检索结果

**清理缓存**：
```bash
# Windows
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse

# Linux/macOS
find . -type d -name __pycache__ -exec rm -r {} +
```

---

## 常见问题

### 安装相关

**Q: uv安装失败怎么办？**

A: 可以使用pip：
```bash
pip install -r requirements.txt
python agno_agent.py
```

**Q: GPU版torch安装失败？**

A: 检查CUDA版本：
```bash
nvidia-smi  # 查看CUDA Version

# 根据版本安装：
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 配置相关

**Q: 找不到.env文件？**

A: 手动创建：
```bash
# 复制示例文件
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS

# 或手动创建并填写内容
```

**Q: API key错误？**

A: 检查：
1. `.env` 文件中的key是否正确
2. key前后没有多余空格
3. 重启服务让配置生效

### 运行相关

**Q: 启动时提示"T5 model not found"？**

A: 
1. 如果没有T5模型，在 `.env` 中设置：
   ```bash
   ENABLE_T5_EVALUATOR=false
   ```
2. 或获取预训练T5模型

**Q: AgentOS连接失败？**

A: 检查：
1. 服务是否正常启动（看到"访问地址"日志）
2. 端口7777是否被占用
3. 防火墙是否阻止
4. 使用 `http://127.0.0.1:7777` 而非 `localhost`

**Q: 查询很慢怎么办？**

A: 
1. 启用GPU加速（性能提升10-40倍）
2. 调整 `T5_BATCH_SIZE` 增加批处理
3. 启用快速路径：`DISABLE_FAST_PATH=false`
4. 减少 `top_k` 值（如改为5）

### 使用相关

**Q: AI回答不准确？**

A: 
1. 确保上传了相关文档
2. 检查文档是否正确解析
3. 调整CRAG阈值（提高 `upper_threshold`）
4. 启用详细日志查看评估分数

**Q: 如何上传大量PDF？**

A: 
1. 使用 `upload_pdf_directory` 工具
2. 在对话中说："上传文件夹 /path/to/pdfs"
3. 或使用API批量上传

**Q: 如何清空知识库？**

A: 
1. 在对话中说："清空知识库"
2. 或手动删除：
   ```bash
   rm -rf tmp/lancedb/*  # 向量数据
   rm tmp/knowledge_contents.db  # 内容数据库
   ```

---

## 项目结构

```
Agno-RAG/
├── agno_agent.py              # 主程序入口
├── rag_tools.py               # RAG工具实现
├── crag_core.py               # CRAG核心组件
├── crag_layer.py              # CRAG评估层
├── document_processor.py      # 文档处理
├── persistent_vector_store.py # 向量存储
├── pyproject.toml             # 项目配置
├── .env                       # 环境变量（需创建）
├── .env.example               # 环境变量模板
├── README.md                  # 本文档
├── finetuned_t5_evaluator/    # T5模型目录
├── tmp/                       # 临时文件
│   ├── lancedb/              # 向量数据库
│   ├── data.db               # Agent数据
│   └── knowledge_contents.db # 知识库内容
└── rag_database.db           # RAG数据库
```

---

## 技术栈

- **框架**：Agno 2.2.13+
- **LLM**：DeepSeek V3.1
- **向量数据库**：LanceDB
- **嵌入模型**：Sentence-Transformers
- **评估模型**：T5 (fine-tuned)
- **Web框架**：FastAPI
- **UI**：AgentOS

---

## 开发路线图

- [ ] 支持更多文档格式（Word、Excel、Markdown）
- [ ] 多语言支持（优化中文处理）
- [ ] 对话历史管理
- [ ] 文档版本控制
- [ ] 批量评估和测试框架
- [ ] Docker部署支持

---

## 许可证

MIT License

---

## 贡献

欢迎提交Issue和Pull Request！

---

## 联系方式

- GitHub Issues: [项目Issues页面]
- 邮箱: [your-email@example.com]

---

## 致谢

- [Agno Framework](https://github.com/agno-agi/agno)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- DeepSeek Team
- SiliconFlow

---

**最后更新**: 2025-11-19

**开始使用**: `uv sync && uv run python agno_agent.py` 🚀

#   L a s t   u p d a t e d :   2 0 2 5 - 1 1 - 1 9  
 