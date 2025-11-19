# Agno-RAG: CRAG-Enhanced Intelligent Document Q&A System

> A high-performance document question-answering system based on the Agno framework and CRAG (Corrective Retrieval Augmented Generation) technology

## Table of Contents

- [Project Introduction](#project-introduction)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [CRAG Strategy Explained](#crag-strategy-explained)
- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Installation Steps](#installation-steps)
  - [Configuration](#configuration)
- [AgentOS Connection](#agentos-connection)
- [Strategy Configuration](#strategy-configuration)
- [Performance Optimization](#performance-optimization)
- [FAQ](#faq)

---

## Project Introduction

Agno-RAG is an intelligent document question-answering system that enables AI to understand and answer questions about your uploaded PDF documents. Unlike traditional RAG systems, this system employs **CRAG (Corrective Retrieval Augmented Generation)** technology, which intelligently evaluates the quality of retrieval results and automatically adopts optimal strategies to significantly improve answer accuracy.

### Use Cases

- Technical documentation queries (standards, specifications, technical manuals)
- Enterprise knowledge base management
- Contract document analysis
- Academic paper reading assistant

---

## Core Features

### 1. Intelligent Document Management

- **PDF Upload and Parsing**: Support for batch PDF document uploads
- **Intelligent Chunking**: Automatically segments documents into semantically coherent chunks
- **Vector Storage**: Efficient vector retrieval using LanceDB
- **Document Tracking**: Support for filtering queries by document ID

### 2. CRAG-Enhanced Retrieval

The core innovation of this system is **CRAG (Corrective RAG)**, which includes three key components:

#### Semantic Retrieval Evaluator (T5-based)

- Uses a fine-tuned T5 model to evaluate semantic relevance between retrieved documents and queries
- Returns continuous scores (-1 to 1), more accurate than simple keyword matching
- **GPU acceleration supported**, providing 10-40x speedup for evaluation

#### Three-Action Routing Mechanism

Automatically selects optimal strategies based on document quality:

| Action | Trigger Condition | Processing Method |
|--------|------------------|-------------------|
| **Correct** | High-quality documents (score > 0.6) | Use retrieval results directly |
| **Incorrect** | Low-quality documents (score < 0.2) | Trigger external knowledge search |
| **Ambiguous** | Medium-quality documents (0.2 ≤ score ≤ 0.6) | Knowledge refinement and reconstruction |

#### Knowledge Refiner

- **Decompose-Reconstruct** strategy: Splits long documents into knowledge strips
- **Semantic Filtering**: Removes content irrelevant to the query
- **Information Reorganization**: Reorganizes knowledge to provide clearer answers

### 3. Query Enhancement Strategies

- **Query Expansion**: Automatically generates similar queries to improve recall
- **Multi-Strategy Retrieval**:
  - Original query retrieval
  - Document ID filtering
  - Hybrid retrieval
- **Result Deduplication and Sorting**: Intelligently merges and sorts retrieval results

### 4. Web Interface (AgentOS Integration)

- Modern web interface
- Conversational interaction
- Visual knowledge base management
- Query history tracking

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
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

### Data Flow

1. **Document Upload** → PDF parsing → Text extraction → Chunking → Vectorization → Storage
2. **User Query** → Vector retrieval → CRAG evaluation → Action routing → Knowledge refinement → LLM answer generation

---

## CRAG Strategy Explained

### Why CRAG?

Problems with traditional RAG systems:
- Does not distinguish quality of retrieval results
- Low-quality documents mislead AI
- Cannot handle missing knowledge base scenarios
- Redundant information interferes with answer generation

CRAG solutions:
- **Intelligent Evaluation**: Automatically determines if retrieval results are reliable
- **Dynamic Routing**: Selects different processing strategies based on quality
- **Knowledge Refinement**: Extracts key information, filters noise
- **External Supplementation**: Triggers web search when quality is poor

### CRAG Workflow

```
Query → Vector Retrieval
         │
         ▼
    CRAG Evaluation
         │
    ┌────┴────┐
    ▼         ▼
Semantic      Fast Path
Evaluation    (Lexical)
(T5 Model)
    │         │
    └────┬────┘
         ▼
    Quality Score (confidence)
         │
    ┌────┼────┐
    ▼    ▼    ▼
 Correct Ambiguous Incorrect
    │    │         │
    │    ▼         ▼
    │  Refine   Web Search
    │    │         │
    └────┴────┬────┘
              ▼
         Generate Answer
```

### Three Evaluation Modes

#### 1. Fast Path
- **Trigger Condition**: High retrieval score (>0.95) and moderate document count
- **Characteristics**: Uses lexical scoring, fast
- **Use Case**: High-confidence queries, seeking speed

#### 2. Full CRAG
- **Trigger Condition**: Default mode or uncertain retrieval quality
- **Characteristics**: Uses T5 semantic evaluation, high accuracy
- **Use Case**: Queries requiring high accuracy

#### 3. Performance Mode
- **Trigger Condition**: Set `DISABLE_FAST_PATH=true`
- **Characteristics**: Forces full CRAG, facilitates performance testing
- **Use Case**: Development debugging, performance optimization

---

## Quick Start

### Requirements

#### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| Memory | 8GB | 16GB+ |
| Storage | 20GB available | 50GB+ SSD |
| GPU | None (use CPU) | NVIDIA GPU 6GB+ VRAM |

**GPU Support**:
- Recommended: NVIDIA GPU (RTX 3060 or higher)
- Performance boost: GPU provides 10-40x speedup for T5 evaluation
- Supported CUDA versions: 11.8 or 12.1

#### Software Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or higher
- **Package Manager**: uv (recommended) or pip

---

### Installation Steps

#### Step 1: Clone the Project

```bash
# Clone repository
git clone https://github.com/Xarmian10/Agno-CRAG.git
cd Agno-CRAG
```

#### Step 2: Install uv (Recommended Package Manager)

**Windows (PowerShell)**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Step 3: Install Dependencies

**Basic Installation (CPU version)**:
```bash
uv sync
```

**GPU Version Installation** (Recommended for better performance):

If you have an NVIDIA GPU, the project is already configured to use the GPU version of PyTorch:

```bash
# 1. First verify GPU is available
nvidia-smi

# 2. Sync installation (GPU index already configured)
uv sync

# 3. Verify GPU support
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note**:
- The project's `pyproject.toml` has PyTorch GPU index configured
- No need to manually specify index URL
- If verification fails, see GPU configuration troubleshooting

---

### Configuration

#### Step 4: Configure Environment Variables

Create `.env` file (copy from template):

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

Then edit the `.env` file:

```bash
# ============================================================
# Core Configuration
# ============================================================

# DeepSeek API Configuration (Required)
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
DEEPSEEK_MODEL_ID=deepseek-ai/DeepSeek-V3.1-Terminus

# ============================================================
# CRAG Configuration
# ============================================================

# Enable Complete CRAG (Recommended)
USE_COMPLETE_CRAG=true

# Enable T5 Semantic Evaluator
ENABLE_T5_EVALUATOR=true

# T5 Model Path
T5_EVALUATOR_PATH=finetuned_t5_evaluator

# T5 Batch Size
# CPU: 4-8, GPU(6GB): 8-12, GPU(8GB): 12-16, GPU(12GB+): 16-24
T5_BATCH_SIZE=12

# Disable Fast Path (Force full CRAG for performance testing)
DISABLE_FAST_PATH=false

# ============================================================
# Web Search Configuration (Optional)
# ============================================================

# Enable Web Search Enhancement
ENABLE_WEB_SEARCH=false

# SerpAPI Key (if web search enabled)
# SERPAPI_KEY=your-serpapi-key

# ============================================================
# Performance Configuration
# ============================================================

# Verbose Logging
VERBOSE_CRAG=false
```

#### Step 5: Obtain API Keys

##### 1. DeepSeek API (Required)

This project uses DeepSeek API provided by SiliconFlow:

1. Visit [SiliconFlow](https://cloud.siliconflow.cn/)
2. Register and log in
3. Go to "API Keys" page
4. Create new API key
5. Copy key to `DEEPSEEK_API_KEY` in `.env` file

**Pricing**:
- New users typically have free credits
- Charged per token, relatively low cost
- See official website for detailed pricing

##### 2. SerpAPI (Optional, for Web Search)

If external knowledge enhancement is needed:

1. Visit [SerpAPI](https://serpapi.com/)
2. Register account
3. Get API key
4. Set in `.env`:
   ```bash
   ENABLE_WEB_SEARCH=true
   SERPAPI_KEY=your-serpapi-key
   ```

#### Step 6: Prepare T5 Model

This system requires a fine-tuned T5 evaluator model:

**Option A: Use Pre-trained Model** (Recommended)

If you have a pre-trained T5 model, place it in the project root:

```
Agno-CRAG/
├── finetuned_t5_evaluator/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── spiece.model
└── ...
```

**Option B: Disable T5 Evaluator** (Fallback mode)

If you don't have the T5 model temporarily, you can use lexical scoring:

Set in `.env`:
```bash
ENABLE_T5_EVALUATOR=false
```

**Note**: Disabling T5 reduces evaluation accuracy, recommended for testing only.

---

## AgentOS Connection

AgentOS is a modern web interface provided by the Agno framework, allowing you to interact with AI through a browser.

### Start Service

```bash
# Run with uv (recommended)
uv run python agno_agent.py

# Or run directly (if virtual environment activated)
python agno_agent.py
```

### Access Interface

After successful startup, you will see:

```
Model configuration loaded: deepseek-ai/DeepSeek-V3.1-Terminus
SQLite database configured
...
Knowledge base created (using LanceDB)
...

Access URL: http://127.0.0.1:7777
API Docs: http://127.0.0.1:7777/docs
```

#### Local Access

Open in browser: `http://127.0.0.1:7777`

You will see Agno's local debug interface.

#### Connect to AgentOS Cloud

1. **Register AgentOS Account**
   - Visit [os.agno.com](https://os.agno.com)
   - Register and log in

2. **Add Agent Connection**
   - Click "Add Agent"
   - Select "Local Agent"
   - Enter connection info:
     ```
     Name: Agno-RAG
     URL: http://127.0.0.1:7777
     ```
   - Click "Connect"

3. **Start Using**
   - After successful connection, you can:
     - Chat with AI
     - Manage documents in "Knowledge" tab
     - View conversation history
     - Adjust Agent settings

### AgentOS Features

#### 1. Chat Interface

- Enter questions, AI automatically calls `query_documents` tool
- Supports multi-turn conversations, AI remembers context
- Can upload new documents at any time

#### 2. Knowledge Base Management

- **View Documents**: Display all uploaded documents
- **Upload Documents**:
  - Click "Upload" button
  - Select PDF files (supports multiple selection)
  - System automatically parses and stores
- **Delete Documents**: Click delete button next to document

#### 3. Tool Invocation

AI automatically calls the following tools:

- `query_documents`: Query document content (automatically uses CRAG)
- `upload_pdf_document`: Upload single PDF
- `list_documents`: List all documents
- `delete_document`: Delete specified document

You can say directly in conversation:
- "Upload this PDF"
- "List all documents"
- "Delete document with ID xxx"

---

## Strategy Configuration

### CRAG Core Parameters

#### 1. Evaluator Configuration

```bash
# Enable/Disable T5 Evaluator
ENABLE_T5_EVALUATOR=true

# T5 Batch Size (affects GPU utilization)
# Recommended values:
#   CPU: 4-8
#   GPU 6GB: 8-12
#   GPU 8GB: 12-16
#   GPU 12GB+: 16-32
T5_BATCH_SIZE=12

# T5 Model Path
T5_EVALUATOR_PATH=finetuned_t5_evaluator
```

#### 2. Action Router Thresholds

Configure in `get_action_router()` function in `rag_tools.py`:

```python
router = CompleteActionRouter(
    evaluator=evaluator,
    web_searcher=web_searcher,
    upper_threshold=0.6,  # Correct threshold (above = high quality)
    lower_threshold=0.2,  # Incorrect threshold (below = low quality)
)
```

**Threshold Adjustment Recommendations**:

| Scenario | upper_threshold | lower_threshold | Description |
|----------|----------------|----------------|-------------|
| Strict Mode | 0.7 | 0.3 | More ambiguous, more refinement |
| Balanced Mode | 0.6 | 0.2 | Default, balances accuracy and performance |
| Relaxed Mode | 0.5 | 0.1 | More correct, faster response |

#### 3. Retrieval Parameters

When querying in conversation, the system defaults to:

```python
query_documents(
    query="Your question",
    top_k=10,        # Number of documents to retrieve
    threshold=0.15,  # Similarity threshold
    mode="excerption" # Knowledge refinement mode
)
```

**Parameter Explanation**:

- **top_k** (5-20):
  - Larger: Higher recall, but slower
  - Smaller: Faster, but may miss relevant documents
  - Recommended: 10

- **threshold** (0.0-1.0):
  - Higher: Only returns high-similarity documents, may have insufficient recall
  - Lower: Returns more documents, may include noise
  - Recommended: 0.15

- **mode** (excerption/original):
  - `excerption`: Enable knowledge refinement (recommended)
  - `original`: Use original documents

#### 4. Fast Path Configuration

```bash
# Disable Fast Path (force full CRAG)
DISABLE_FAST_PATH=false

# Fast Path trigger conditions (in crag_layer.py)
FAST_PATH_SCORE_THRESHOLD=0.95  # Retrieval score threshold
FAST_PATH_MAX_DOCS=15           # Maximum document count
```

### Web Search Configuration

```bash
# Enable Web Search
ENABLE_WEB_SEARCH=true

# SerpAPI Configuration
SERPAPI_KEY=your-key

# Web search parameters (in rag_tools.py)
WEB_SEARCH_NUM_RESULTS=5  # Number of search results
```

### Logging and Debugging

```bash
# Enable verbose logging
VERBOSE_CRAG=true

# Logs will show:
# - Retrieval process details
# - CRAG evaluation scores
# - Action routing decisions
# - Knowledge refinement results
# - Performance statistics
```

---

## Performance Optimization

### GPU Acceleration Configuration

#### Check GPU Support

```bash
# Check GPU
nvidia-smi

# Verify PyTorch GPU support
uv run python -c "import torch; print(torch.cuda.is_available())"
```

#### GPU Configuration

The project is already configured for GPU support. If issues occur:

1. **Verify GPU version torch is installed**:
   ```bash
   uv pip list | grep torch
   # Should show: torch 2.5.1+cu121
   ```

2. **If CPU version is shown**:
   ```bash
   # Reinstall
   uv pip uninstall torch
   uv sync
   ```

3. **Adjust batch size**:
   ```bash
   # In .env, adjust based on GPU memory
   T5_BATCH_SIZE=16  # For 8GB VRAM
   ```

### Performance Benchmarks

| Configuration | 10 Doc Evaluation Time | Throughput | Speedup |
|---------------|----------------------|------------|---------|
| CPU (i7) | ~30-40 sec | ~3 docs/sec | Baseline |
| GPU (RTX 3060 6GB) | ~2-3 sec | ~30 docs/sec | 10x |
| GPU (RTX 3070 8GB) | ~1-2 sec | ~50 docs/sec | 15x |
| GPU (RTX 4070 12GB) | ~0.5-1 sec | ~100 docs/sec | 30x |

### Cache Optimization

System automatically caches:
- Document vectors
- T5 evaluator instance
- Retrieval results

**Clear cache**:
```bash
# Windows
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse

# Linux/macOS
find . -type d -name __pycache__ -exec rm -r {} +
```

---

## FAQ

### Installation

**Q: What if uv installation fails?**

A: You can use pip:
```bash
pip install -r requirements.txt
python agno_agent.py
```

**Q: GPU version torch installation fails?**

A: Check CUDA version:
```bash
nvidia-smi  # Check CUDA Version

# Install based on version:
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Configuration

**Q: Cannot find .env file?**

A: Create manually:
```bash
# Copy example file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS

# Or create manually and fill in content
```

**Q: API key error?**

A: Check:
1. Key in `.env` file is correct
2. No extra spaces before or after key
3. Restart service for config to take effect

### Runtime

**Q: Startup shows "T5 model not found"?**

A: 
1. If no T5 model, set in `.env`:
   ```bash
   ENABLE_T5_EVALUATOR=false
   ```
2. Or obtain pre-trained T5 model

**Q: AgentOS connection failed?**

A: Check:
1. Service started normally (see "Access URL" in logs)
2. Port 7777 not occupied
3. Firewall not blocking
4. Use `http://127.0.0.1:7777` instead of `localhost`

**Q: Query is slow?**

A: 
1. Enable GPU acceleration (10-40x performance improvement)
2. Adjust `T5_BATCH_SIZE` to increase batch processing
3. Enable fast path: `DISABLE_FAST_PATH=false`
4. Reduce `top_k` value (e.g., change to 5)

### Usage

**Q: AI answers are inaccurate?**

A: 
1. Ensure relevant documents are uploaded
2. Check if documents are parsed correctly
3. Adjust CRAG threshold (increase `upper_threshold`)
4. Enable verbose logging to view evaluation scores

**Q: How to upload many PDFs?**

A: 
1. Use `upload_pdf_directory` tool
2. Say in conversation: "Upload folder /path/to/pdfs"
3. Or use API for batch upload

**Q: How to clear knowledge base?**

A: 
1. Say in conversation: "Clear knowledge base"
2. Or delete manually:
   ```bash
   rm -rf tmp/lancedb/*  # Vector data
   rm tmp/knowledge_contents.db  # Content database
   ```

---

## Project Structure

```
Agno-CRAG/
├── agno_agent.py              # Main program entry
├── rag_tools.py               # RAG tool implementation
├── crag_core.py               # CRAG core components
├── crag_layer.py              # CRAG evaluation layer
├── document_processor.py      # Document processing
├── persistent_vector_store.py # Vector storage
├── pyproject.toml             # Project configuration
├── .env                       # Environment variables (needs creation)
├── .env.example               # Environment variable template
├── README.md                  # This document
├── finetuned_t5_evaluator/    # T5 model directory
├── tmp/                       # Temporary files
│   ├── lancedb/              # Vector database
│   ├── data.db               # Agent data
│   └── knowledge_contents.db # Knowledge base content
└── rag_database.db           # RAG database
```

---

## Tech Stack

- **Framework**: Agno 2.2.13+
- **LLM**: DeepSeek V3.1
- **Vector Database**: LanceDB
- **Embedding Model**: Sentence-Transformers
- **Evaluation Model**: T5 (fine-tuned)
- **Web Framework**: FastAPI
- **UI**: AgentOS

---

## Roadmap

- Support for more document formats (Word, Excel, Markdown)
- Multi-language support (optimize Chinese processing)
- Conversation history management
- Document version control
- Batch evaluation and testing framework
- Docker deployment support

---

## License

MIT License

---

## Contributing

Issues and Pull Requests are welcome!

---

## Contact

- GitHub Issues: [Project Issues Page](https://github.com/Xarmian10/Agno-CRAG/issues)
- Email: contact@example.com

---

## Acknowledgments

- [Agno Framework](https://github.com/agno-agi/agno)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- DeepSeek Team
- SiliconFlow

---

**Last Updated**: 2025-11-19

**Get Started**: `uv sync && uv run python agno_agent.py`
