import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try to load from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, use system environment variables
    pass

# Configure root logger to ensure all logs are visible
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr to avoid buffering
    ],
    force=True,  # Override any existing configuration
)

# Safe stderr writing function (defined early to avoid NameError)
def safe_write_stderr(message: str):
    """Safely write to stderr handling encoding issues."""
    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            safe_message = message.encode('ascii', errors='replace').decode('ascii', errors='replace')
            sys.stderr.write(safe_message)
            sys.stderr.flush()
        except Exception:
            try:
                sys.stderr.buffer.write(message.encode('utf-8', errors='replace'))
                sys.stderr.buffer.flush()
            except Exception:
                pass

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.models.openai import OpenAILike
from typing import Optional

from rag_tools import RAGTools

# Try to import Agno knowledge base classes
try:
    from agno.knowledge.knowledge import Knowledge
    try:
        from agno.vectordb.lancedb import LanceDb
        VECTOR_DB_TYPE = "lancedb"
    except ImportError:
        try:
            from agno.vectordb.pgvector import PgVector
            VECTOR_DB_TYPE = "pgvector"
        except ImportError:
            VECTOR_DB_TYPE = None
    # Try to import embedders
    try:
        from agno.knowledge.embedder.openai import OpenAIEmbedder
        OPENAI_EMBEDDER_AVAILABLE = True
    except ImportError:
        OPENAI_EMBEDDER_AVAILABLE = False
        OpenAIEmbedder = None
    
    try:
        from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
        SENTENCE_TRANSFORMER_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMER_AVAILABLE = False
        SentenceTransformerEmbedder = None
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    Knowledge = None
    LanceDb = None
    PgVector = None
    OpenAIEmbedder = None
    SentenceTransformerEmbedder = None
    VECTOR_DB_TYPE = None
    OPENAI_EMBEDDER_AVAILABLE = False
    SENTENCE_TRANSFORMER_AVAILABLE = False


# Load model configuration from environment variables
# These can be set in .env file or system environment variables
deepseek_model_id = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-V3.1-Terminus")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")

# Validate that API key is provided
if not deepseek_api_key:
    safe_write_stderr("警告: DEEPSEEK_API_KEY 未设置。请检查 .env 文件或环境变量。\n")
    safe_write_stderr("提示: 复制 .env.example 到 .env 并填写您的 API key。\n")
    raise ValueError(
        "DEEPSEEK_API_KEY is required. "
        "Please set it in .env file or as an environment variable. "
        "See .env.example for reference."
    )

deepseek_model = OpenAILike(
    id=deepseek_model_id,
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
)
safe_write_stderr(f"模型配置已加载: {deepseek_model_id}\n")

# Setup the SQLite database for Agent
# This database stores agent sessions, history, and other agent data
db_path = Path("tmp/data.db")
db_path.parent.mkdir(parents=True, exist_ok=True)
db = SqliteDb(db_file=str(db_path))
safe_write_stderr("SQLite 数据库已配置\n")
safe_write_stderr(f"  数据库路径: {db_path.absolute()}\n")

# Initialize RAG tools
rag_tools = RAGTools(db_path="rag_database.db")

# Set knowledge base in rag_tools (will be set after knowledge_base is created)
# This allows CRAG to use Agno Knowledge instead of old vector store

# Log agent initialization
safe_write_stderr("正在初始化 RAG 工具...\n")
safe_write_stderr(f"可用工具数量: {len(rag_tools.tools)}\n")

# Create a knowledge base for AgentOS
# This allows AgentOS to recognize the knowledge base in the UI
knowledge_base = None
safe_write_stderr("正在检查知识库模块...\n")
safe_write_stderr(f"  KNOWLEDGE_BASE_AVAILABLE: {KNOWLEDGE_BASE_AVAILABLE}\n")
safe_write_stderr(f"  Knowledge class: {Knowledge is not None}\n")
safe_write_stderr(f"  VECTOR_DB_TYPE: {VECTOR_DB_TYPE}\n")

if KNOWLEDGE_BASE_AVAILABLE and Knowledge and VECTOR_DB_TYPE:
    try:
        from pathlib import Path
        
        if VECTOR_DB_TYPE == "lancedb":
            # Use LanceDB (no PostgreSQL required)
            lancedb_path = Path("tmp/lancedb")
            lancedb_path.mkdir(parents=True, exist_ok=True)
            
            # Configure embedder - use SentenceTransformer (local, no API key needed)
            # or OpenAIEmbedder with SiliconFlow if available
            embedder = None
            if SENTENCE_TRANSFORMER_AVAILABLE and SentenceTransformerEmbedder:
                # Use local SentenceTransformer model (no API key needed)
                # This is better for Chinese text and doesn't require API calls
                try:
                    embedder = SentenceTransformerEmbedder(
                        id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                        # This model supports both English and Chinese
                    )
                    safe_write_stderr("  嵌入器: SentenceTransformer (本地模型，支持中英文)\n")
                    safe_write_stderr(f"  模型: paraphrase-multilingual-MiniLM-L12-v2\n")
                except Exception as e:
                    safe_write_stderr(f"  警告: SentenceTransformer 初始化失败: {str(e)}\n")
                    safe_write_stderr(f"  错误详情: {type(e).__name__}: {str(e)}\n")
                    embedder = None
            
            if embedder is None and OPENAI_EMBEDDER_AVAILABLE and OpenAIEmbedder:
                # Fallback: Try to use OpenAIEmbedder with SiliconFlow
                # Note: This may not work if SiliconFlow doesn't support embeddings
                try:
                    # Check if we can use SiliconFlow for embeddings
                    # Most likely this won't work, so we'll skip it
                    safe_write_stderr("  警告: 未配置嵌入器，将使用默认 OpenAIEmbedder（需要 OpenAI API key）\n")
                    safe_write_stderr("  建议: 安装 sentence-transformers 以使用本地嵌入模型\n")
                except Exception:
                    pass
            
            # Create vector database
            # Note: embedder can be passed to vector_db or Knowledge
            vector_db_kwargs = {
                "table_name": "rag_documents",
                "uri": str(lancedb_path.absolute())
            }
            # Some vector databases accept embedder parameter
            if embedder:
                vector_db_kwargs["embedder"] = embedder
            
            vector_db = LanceDb(**vector_db_kwargs)
            
            # Set up ContentsDB - tracks content metadata
            # Using SQLite for development (as shown in the example)
            contents_db_path = Path("tmp/knowledge_contents.db")
            contents_db_path.parent.mkdir(parents=True, exist_ok=True)
            # SQLite for development (knowledge_table is for PostgresDb, not needed for SqliteDb)
            contents_db = SqliteDb(
                db_file=str(contents_db_path)
            )
            # Ensure ContentsDB is initialized by checking/creating the database file
            # This ensures AgentOS can detect the knowledge base
            try:
                import sqlite3
                test_conn = sqlite3.connect(str(contents_db_path), check_same_thread=False)
                test_conn.close()
                safe_write_stderr(f"  ContentsDB 已初始化: {contents_db_path.absolute()}\n")
            except Exception as e:
                safe_write_stderr(f"  [警告] ContentsDB 初始化检查失败: {str(e)}\n")
            
            # Configure PDFReader for Knowledge to handle PDF files uploaded via AgentOS
            # This ensures PDFs uploaded through AgentOS are properly processed
            try:
                from agno.knowledge.reader.pdf_reader import PDFReader
                from agno.knowledge.chunking.document import DocumentChunking
                
                # Create a PDF reader with appropriate chunking strategy
                pdf_reader = PDFReader(
                    split_on_pages=True,  # Split by pages for better context
                    chunking_strategy=DocumentChunking(chunk_size=4000),  # Reasonable chunk size
                )
                
                # Configure readers dictionary for Knowledge
                # This tells Knowledge how to process PDF files
                readers = {
                    "pdf": pdf_reader,
                    ".pdf": pdf_reader,
                }
                
                safe_write_stderr("  PDFReader 已配置，支持通过 AgentOS 上传 PDF 文件\n")
            except Exception as e:
                safe_write_stderr(f"  警告: PDFReader 配置失败: {str(e)}\n")
                safe_write_stderr(f"  建议: 确保已安装 pypdf: uv pip install pypdf\n")
                readers = None
            
            # Create knowledge base with name, vector_db, contents_db, and readers
            # The name will be displayed in AgentOS Knowledge section
            # contents_db enables content tracking!
            # readers enables automatic PDF processing for files uploaded via AgentOS
            # Note: embedder is passed to vector_db (LanceDb), not directly to Knowledge
            knowledge_base_kwargs = {
                "name": "RAG Knowledge Base",
                "vector_db": vector_db,  # embedder is already configured in vector_db
                "contents_db": contents_db,  # This enables content tracking!
                "max_results": 10
            }
            
            # Add readers if configured
            if readers:
                knowledge_base_kwargs["readers"] = readers
            
            knowledge_base = Knowledge(**knowledge_base_kwargs)
            
            # Log embedder status
            if embedder:
                safe_write_stderr(f"  嵌入器已配置: {embedder.__class__.__name__} (通过 vector_db)\n")
            else:
                safe_write_stderr(f"  警告: 未配置嵌入器，vector_db 将使用默认 OpenAIEmbedder（需要 OpenAI API key）\n")
                safe_write_stderr(f"  建议: 运行 'uv pip install sentence-transformers' 安装本地嵌入模型\n")
            safe_write_stderr(f"知识库已创建（使用 LanceDB）\n")
            safe_write_stderr(f"  知识库名称: {knowledge_base.name}\n")
            safe_write_stderr(f"  向量数据库: {vector_db.__class__.__name__}\n")
            safe_write_stderr(f"  向量存储路径: {lancedb_path.absolute()}\n")
            safe_write_stderr(f"  内容数据库: SQLite ({contents_db_path.absolute()})\n")
            safe_write_stderr(f"  内容跟踪: 已启用\n")
            
            # Set knowledge base in rag_tools so CRAG can use it
            from rag_tools import set_knowledge_base
            set_knowledge_base(knowledge_base)
            safe_write_stderr("  CRAG 已配置为使用 Agno Knowledge 数据库\n")
            
            # Store knowledge_base for async sync after FastAPI startup
            # We'll sync documents in the FastAPI startup event to avoid event loop conflicts
            _knowledge_base_to_sync = knowledge_base
            safe_write_stderr("知识库已准备就绪，将在服务启动后同步内容\n")
                
        elif VECTOR_DB_TYPE == "pgvector":
            # Use PgVector (requires PostgreSQL)
            # Uncomment and configure if you have PostgreSQL
            # knowledge_base = Knowledge(
            #     name="RAG Knowledge Base",
            #     vector_db=PgVector(
            #         table_name="rag_documents",
            #         db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"
            #     ),
            #     max_results=10
            # )
            safe_write_stderr("PgVector 可用，但需要 PostgreSQL 配置（当前使用 LanceDB）\n")
            knowledge_base = None
        else:
            safe_write_stderr("向量数据库不可用（不影响 RAG 功能）\n")
            knowledge_base = None
            
    except Exception as e:
        safe_write_stderr(f"[错误] 初始化知识库时出错: {str(e)}\n")
        import traceback
        safe_write_stderr(f"[错误详情]\n{traceback.format_exc()}\n")
        knowledge_base = None
else:
    if not KNOWLEDGE_BASE_AVAILABLE:
        safe_write_stderr("[警告] 知识库模块导入失败\n")
    elif not Knowledge:
        safe_write_stderr("[警告] Knowledge 类不可用\n")
    elif not VECTOR_DB_TYPE:
        safe_write_stderr("[警告] 向量数据库不可用（需要 LanceDB 或 PgVector）\n")
    else:
        safe_write_stderr("[警告] 知识库模块不可用（不影响 RAG 功能）\n")

# Create the Agent with RAG tools and knowledge base
# Setup a basic agent with the SQLite database
agent_kwargs = {
    "name": "Agno Agent",
    "model": deepseek_model,
    "db": db,  # Use the configured SQLite database
    "tools": rag_tools.tools,
    "add_history_to_context": True,
    "markdown": True,
    "instructions": (
        "You are a helpful assistant with access to a knowledge base of PDF documents. "
        "When users ask questions, use the 'query_documents' tool to search the knowledge base. "
        "If relevant information is found, use it to answer the question. "
        "\n\n"
        "Document Management Tools:\n"
        "- 'upload_pdf_document': Upload a single PDF file to the knowledge base. Requires file_path parameter.\n"
        "- 'upload_pdf_directory': Upload all PDF files from a directory. Requires directory path. "
        "  Optional: pattern (default: '*.pdf'), recursive (default: False).\n"
        "- 'list_documents': List all documents in the knowledge base.\n"
        "- 'delete_document': Delete a specific document by doc_id.\n"
        "- 'clear_knowledge_base': Clear all documents from the knowledge base (use with caution!).\n"
        "\n"
        "When users want to upload a single document, use 'upload_pdf_document' with the file path. "
        "When users want to upload multiple PDFs from a folder, use 'upload_pdf_directory' with the directory path. "
        "When users want to clear the database, use the 'clear_knowledge_base' tool."
    ),
}

# Add knowledge base if available
# ContentsDB is mandatory for AgentOS Knowledge management interface
# This makes the knowledge base visible in AgentOS Knowledge section
if knowledge_base:
    # Verify contents_db is configured (required for AgentOS)
    if hasattr(knowledge_base, 'contents_db') and knowledge_base.contents_db:
        agent_kwargs["knowledge"] = knowledge_base
        agent_kwargs["search_knowledge"] = True  # Enable automatic knowledge search
        safe_write_stderr(f"知识库已添加到 Agent: {knowledge_base.name}\n")
        safe_write_stderr(f"  自动搜索: 已启用\n")
        safe_write_stderr(f"  ContentsDB: 已配置（AgentOS Knowledge 页面可用）\n")
    else:
        safe_write_stderr("警告: 知识库缺少 ContentsDB，AgentOS Knowledge 页面可能不可用\n")
        # Still add knowledge base even without contents_db for basic functionality
        agent_kwargs["knowledge"] = knowledge_base
        agent_kwargs["search_knowledge"] = True
else:
    safe_write_stderr("警告: 知识库未添加到 Agent，AgentOS Knowledge 部分可能不可用\n")

agno_agent = Agent(**agent_kwargs)


# Create the AgentOS
# ContentsDB is required for AgentOS Knowledge page
# Important: Do NOT create a new FastAPI app, use the one from agent_os.get_app()
safe_write_stderr("正在创建 AgentOS...\n")
agent_os = AgentOS(
    name="Agno RAG Agent",
    description="RAG-enabled agent with knowledge base capabilities",
    id="agno-rag-agent",
    agents=[agno_agent]
)
# Get the FastAPI app for the AgentOS
app = agent_os.get_app()
safe_write_stderr("AgentOS 已创建，FastAPI 应用已初始化\n")
if knowledge_base and hasattr(knowledge_base, 'contents_db') and knowledge_base.contents_db:
    safe_write_stderr("  ContentsDB 已配置，AgentOS Knowledge 页面可用\n")
else:
    safe_write_stderr("  警告: ContentsDB 未配置，AgentOS Knowledge 页面可能不可用\n")

# Add CORS middleware to allow connections from Agno OS
# IMPORTANT: Add CORS middleware AFTER getting the app from agent_os.get_app()
# This ensures CORS is properly configured for AgentOS endpoints
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://os.agno.com",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:7777",
        "http://localhost:7777",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)
safe_write_stderr("CORS 中间件已配置，允许来自 os.agno.com 的连接\n")
safe_write_stderr("  CORS 允许的来源: https://os.agno.com, http://localhost:3000, http://localhost:3001, http://127.0.0.1:7777, http://localhost:7777\n")

# Add startup event to sync knowledge base content asynchronously
if knowledge_base:
    @app.on_event("startup")
    async def sync_knowledge_base_on_startup():
        """Sync RAG database content to Agno Knowledge after FastAPI startup."""
        try:
            from rag_tools import get_vector_store
            vector_store = get_vector_store()
            documents = vector_store.list_documents()
            
            if documents:
                safe_write_stderr(f"[启动后] 发现 {len(documents)} 个现有文档，正在同步到 Agno Knowledge...\n")
                
                total_passages = 0
                for doc in documents:
                    try:
                        doc_id = doc.get('doc_id', '')
                        if not doc_id:
                            continue
                        
                        # Get all passages for this document
                        passages = vector_store.get_passages_by_doc(doc_id)
                        
                        if passages:
                            # Combine passages into document text
                            doc_text = "\n\n".join([
                                f"[Page {p.page if p.page else 'N/A'}]\n{p.text}"
                                for p in passages
                            ])
                            
                            # Use async method to add content
                            await knowledge_base.add_content_async(
                                text_content=doc_text,
                                metadata={
                                    "doc_id": doc_id,
                                    "file_path": doc.get('file_path', ''),
                                    "source": "rag_database"
                                }
                            )
                            total_passages += len(passages)
                            safe_write_stderr(f"  [启动后] 已同步文档: {doc_id} ({len(passages)} 个段落)\n")
                    except Exception as e:
                        safe_write_stderr(f"  [启动后] 同步文档 {doc.get('doc_id', 'unknown')} 时出错: {str(e)}\n")
                        continue
                
                if total_passages > 0:
                    safe_write_stderr(f"[启动后] 知识库内容已同步: {total_passages} 个段落来自 {len(documents)} 个文档\n")
                else:
                    safe_write_stderr("[启动后] 警告: 未找到可同步的段落内容\n")
            else:
                safe_write_stderr("[启动后] 未发现现有文档，添加占位符内容以确保 AgentOS 识别知识库...\n")
                # Add a placeholder text so AgentOS can recognize the knowledge base
                # This is critical for AgentOS to show the knowledge base in the UI
                try:
                    await knowledge_base.add_content_async(
                        text_content="RAG Knowledge Base is ready. Upload PDF documents to add content.",
                        metadata={"type": "placeholder", "source": "system"}
                    )
                    safe_write_stderr("[启动后] 占位符内容已添加（AgentOS 应能识别知识库）\n")
                    
                    # Verify content was added
                    try:
                        content_list, total_count = knowledge_base.get_content()
                        safe_write_stderr(f"[启动后] 验证: 知识库中有 {total_count} 项内容\n")
                    except Exception as verify_e:
                        safe_write_stderr(f"[启动后] 验证失败: {str(verify_e)}\n")
                except Exception as e:
                    safe_write_stderr(f"[启动后] 添加占位符内容失败: {str(e)}\n")
                    import traceback
                    safe_write_stderr(f"[启动后] 错误详情: {traceback.format_exc()}\n")
                
        except Exception as e:
            safe_write_stderr(f"[启动后] 同步知识库内容时出错: {str(e)}\n")
            import traceback
            safe_write_stderr(f"[启动后] 错误详情: {traceback.format_exc()}\n")

# Add file upload endpoint for PDF documents
from fastapi import UploadFile, File, HTTPException
import tempfile
import shutil

@app.post("/api/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...), doc_id: Optional[str] = None):
    """
    Upload a PDF file to the knowledge base via HTTP endpoint.
    This integrates with Agno's file upload capabilities.
    """
    from rag_tools import _upload_pdf_document_impl
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file to temporary location
    temp_dir = Path(tempfile.gettempdir()) / "agno_rag_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / file.filename
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        result = _upload_pdf_document_impl(
            file_path=str(temp_file_path),
            doc_id=doc_id
        )
        
        # Clean up temporary file
        try:
            temp_file_path.unlink()
        except Exception:
            pass
        
        if result.get("success"):
            return {
                "success": True,
                "message": "PDF uploaded and processed successfully",
                "doc_id": result.get("doc_id"),
                "passage_count": result.get("passage_count"),
                "pages_with_content": result.get("pages_with_content", 0),
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to process PDF")
            )
    except Exception as e:
        # Clean up on error
        try:
            if temp_file_path.exists():
                temp_file_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

# Startup log messages
safe_write_stderr("\n" + "="*60 + "\n")
safe_write_stderr("Agno Agent 已启动，日志系统已配置\n")
safe_write_stderr("="*60 + "\n")
safe_write_stderr("检索日志将输出到此控制台\n")
safe_write_stderr("="*60 + "\n")
safe_write_stderr(f"\nAgentOS 状态:\n")
safe_write_stderr(f"  - Agent 名称: {agno_agent.name}\n")
safe_write_stderr(f"  - 数据库: SQLite ({db_path.absolute()})\n")
safe_write_stderr(f"  - 工具数量: {len(agno_agent.tools) if hasattr(agno_agent, 'tools') else 'N/A'}\n")
safe_write_stderr(f"  - 知识库: {'已配置' if knowledge_base else '未配置'}\n")
if knowledge_base:
    try:
        content_list, total_count = knowledge_base.get_content()
        safe_write_stderr(f"  - 知识库名称: {knowledge_base.name}\n")
        safe_write_stderr(f"  - 知识库内容: {total_count} 项\n")
        if total_count > 0:
            safe_write_stderr(f"  - 知识库状态: 就绪（可在 AgentOS Knowledge 部分查看）\n")
        else:
            safe_write_stderr(f"  - 知识库状态: 空（启动后将添加占位符内容以确保 AgentOS 识别）\n")
        
        # Verify ContentsDB is accessible
        if hasattr(knowledge_base, 'contents_db') and knowledge_base.contents_db:
            safe_write_stderr(f"  - ContentsDB: 已配置（AgentOS Knowledge 页面应可用）\n")
            # Try to verify database file exists
            try:
                if hasattr(knowledge_base.contents_db, 'db_file'):
                    db_file = Path(knowledge_base.contents_db.db_file)
                    if db_file.exists():
                        safe_write_stderr(f"  - ContentsDB 文件: 存在 ({db_file.absolute()})\n")
                    else:
                        safe_write_stderr(f"  - ContentsDB 文件: 不存在 ({db_file.absolute()})\n")
            except Exception:
                pass
        else:
            safe_write_stderr(f"  - ContentsDB: 未配置（AgentOS Knowledge 页面可能不可用）\n")
    except Exception as e:
        safe_write_stderr(f"  - 知识库内容: 无法获取 ({str(e)})\n")
safe_write_stderr(f"  - FastAPI 应用: {'已创建' if app else '未创建'}\n")
safe_write_stderr(f"\n访问地址: http://127.0.0.1:7777\n")
safe_write_stderr(f"API 文档: http://127.0.0.1:7777/docs\n")
safe_write_stderr("="*60 + "\n\n")