"""
Knowledge Agent - Standalone agent for testing Agno Knowledge functionality.

This script creates a knowledge-powered agent that can search through
documents stored in the knowledge base.
"""
import os
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.models.openai import OpenAILike

# Load model configuration from environment variables (same as agno_agent.py)
deepseek_model_id = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-V3.1-Terminus")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")

# Validate that API key is provided
if not deepseek_api_key:
    print("警告: DEEPSEEK_API_KEY 未设置。请检查 .env 文件或环境变量。")
    print("提示: 复制 .env.example 到 .env 并填写您的 API key。")
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

# Set up knowledge base with LanceDB
lancedb_path = Path("tmp/knowledge_lancedb")
lancedb_path.mkdir(parents=True, exist_ok=True)

knowledge = Knowledge(
    name="RAG Knowledge Base",
    vector_db=LanceDb(
        table_name="knowledge_documents",
        uri=str(lancedb_path.absolute())
    ),
    max_results=10
)

print("=" * 60)
print("Knowledge Agent 初始化")
print("=" * 60)
print(f"知识库路径: {lancedb_path.absolute()}")
print(f"模型: {deepseek_model.id}")
print()

# Option 1: Add content from existing RAG database
def sync_from_rag_database():
    """Sync content from existing RAG database to Agno Knowledge."""
    try:
        from rag_tools import get_vector_store
        vector_store = get_vector_store()
        documents = vector_store.list_documents()
        
        if documents:
            print(f"发现 {len(documents)} 个现有文档，正在同步到 Agno Knowledge...")
            
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
                        
                        # Add document content to knowledge base
                        knowledge.add_content(
                            text_content=doc_text,
                            metadata={
                                "doc_id": doc_id,
                                "file_path": doc.get('file_path', ''),
                                "source": "rag_database"
                            }
                        )
                        total_passages += len(passages)
                        print(f"  ✓ 已同步文档: {doc_id} ({len(passages)} 个段落)")
                except Exception as e:
                    print(f"  ✗ 同步文档 {doc.get('doc_id', 'unknown')} 时出错: {str(e)}")
                    continue
            
            if total_passages > 0:
                print(f"\n✓ 知识库内容已同步: {total_passages} 个段落来自 {len(documents)} 个文档")
            else:
                print("\n⚠ 警告: 未找到可同步的段落内容")
            return True
        else:
            print("未发现现有文档")
            return False
    except Exception as e:
        print(f"同步文档时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False

# Option 2: Add content from file or directory
def add_content_from_path(path: str):
    """Add content from a file or directory to the knowledge base."""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"错误: 路径不存在: {path}")
            return False
        
        if path_obj.is_file():
            print(f"正在添加文件: {path}")
            knowledge.add_content(path=str(path_obj.absolute()))
            print(f"✓ 文件已添加: {path}")
        elif path_obj.is_dir():
            print(f"正在添加目录: {path}")
            knowledge.add_content(path=str(path_obj.absolute()))
            print(f"✓ 目录已添加: {path}")
        else:
            print(f"错误: 无效的路径: {path}")
            return False
        
        return True
    except Exception as e:
        print(f"添加内容时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False

# Option 3: Add text content directly
def add_text_content(text: str, metadata: dict = None):
    """Add text content directly to the knowledge base."""
    try:
        knowledge.add_content(
            text_content=text,
            metadata=metadata or {}
        )
        print("✓ 文本内容已添加")
        return True
    except Exception as e:
        print(f"添加文本内容时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False

# Create the agent with knowledge
agent = Agent(
    name="Knowledge Agent",
    model=deepseek_model,
    knowledge=knowledge,
    search_knowledge=True,  # Enable automatic knowledge search
    instructions=[
        "You are a helpful assistant with access to a knowledge base.",
        "Always search your knowledge base before answering questions.",
        "Include source references in your responses when possible.",
        "If you cannot find relevant information in the knowledge base, say so clearly."
    ],
    markdown=True,
)

print("=" * 60)
print("Agent 已创建")
print("=" * 60)
print(f"Agent 名称: {agent.name}")
print(f"知识库: {'已配置' if knowledge else '未配置'}")
print()

# Main function for testing
def main():
    """Main function to interact with the knowledge agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Agent - Test Agno Knowledge functionality")
    parser.add_argument(
        "--sync-rag",
        action="store_true",
        help="Sync content from existing RAG database"
    )
    parser.add_argument(
        "--add-path",
        type=str,
        help="Add content from a file or directory path"
    )
    parser.add_argument(
        "--add-text",
        type=str,
        help="Add text content directly"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query the knowledge base and get a response"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode"
    )
    
    args = parser.parse_args()
    
    # Sync from RAG database if requested
    if args.sync_rag:
        print("\n[选项] 从 RAG 数据库同步内容")
        print("-" * 60)
        sync_from_rag_database()
        print()
    
    # Add content from path if provided
    if args.add_path:
        print(f"\n[选项] 添加路径内容: {args.add_path}")
        print("-" * 60)
        add_content_from_path(args.add_path)
        print()
    
    # Add text content if provided
    if args.add_text:
        print("\n[选项] 添加文本内容")
        print("-" * 60)
        add_text_content(args.add_text)
        print()
    
    # Query if provided
    if args.query:
        print(f"\n[查询] {args.query}")
        print("-" * 60)
        agent.print_response(args.query, stream=True)
        print()
    
    # Interactive mode
    if args.interactive or (not args.sync_rag and not args.add_path and not args.add_text and not args.query):
        print("\n" + "=" * 60)
        print("交互模式")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'sync' 从 RAG 数据库同步内容")
        print("输入 'add <path>' 添加文件或目录")
        print("输入 'status' 查看知识库状态")
        print("-" * 60)
        print()
        
        while True:
            try:
                user_input = input("你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if user_input.lower() == 'sync':
                    sync_from_rag_database()
                    print()
                    continue
                
                if user_input.lower().startswith('add '):
                    path = user_input[4:].strip()
                    add_content_from_path(path)
                    print()
                    continue
                
                if user_input.lower() == 'status':
                    try:
                        content_list, total_count = knowledge.get_content()
                        print(f"知识库内容: {total_count} 项")
                        for i, content in enumerate(content_list[:10], 1):
                            print(f"  {i}. {content.name if hasattr(content, 'name') else content.id}")
                        if total_count > 10:
                            print(f"  ... 还有 {total_count - 10} 项")
                    except Exception as e:
                        print(f"获取知识库状态时出错: {str(e)}")
                    print()
                    continue
                
                # Query the agent
                print("\nAgent: ", end="", flush=True)
                agent.print_response(user_input, stream=True)
                print()
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}")
                import traceback
                print(traceback.format_exc())
                print()

if __name__ == "__main__":
    main()

