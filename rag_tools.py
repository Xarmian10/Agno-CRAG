"""
Agno tools for RAG functionality: upload PDFs and query documents.

This module integrates the complete CRAG system with:
- Semantic retrieval evaluator (T5-based)
- External web search augmenter
- Complete action router
- Enhanced knowledge refiner
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from agno.tools import tool

# Unified logging function that ensures output is visible
# Only writes to stderr directly to avoid duplicate output from logger propagation
def log_to_console(message: str, flush: bool = True):
    """
    Log message to stderr to ensure visibility.
    Handles encoding issues on Windows by using UTF-8 with error handling.
    Note: Only writes to stderr directly to avoid duplicate output from logger propagation.
    """
    # Ensure message is a string
    if not isinstance(message, str):
        message = str(message)
    
    # Write to stderr directly (most reliable, avoids duplicate output)
    try:
        # Try direct write first
        sys.stderr.write(message + "\n")
        if flush:
            sys.stderr.flush()
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If encoding fails, replace problematic characters
        try:
            # Replace emoji and special characters with ASCII equivalents
            safe_message = message.encode('ascii', errors='replace').decode('ascii', errors='replace')
            sys.stderr.write(safe_message + "\n")
            if flush:
                sys.stderr.flush()
        except Exception:
            # Last resort: write raw bytes
            try:
                sys.stderr.buffer.write((message + "\n").encode('utf-8', errors='replace'))
                if flush:
                    sys.stderr.buffer.flush()
            except Exception:
                pass
    except Exception:
        pass

from persistent_vector_store import PersistentVectorStore
from document_processor import DocumentProcessor, Passage
from crag_layer import crag_evaluate_and_route

# Import CRAG core components
try:
    from crag_core import (
        SemanticRetrievalEvaluator,
        WebSearchAugmenter,
        CompleteActionRouter,
    )
    SEMANTIC_EVALUATOR_AVAILABLE = True
    WEB_SEARCH_AVAILABLE = True
    ACTION_ROUTER_AVAILABLE = True
except ImportError:
    SEMANTIC_EVALUATOR_AVAILABLE = False
    WEB_SEARCH_AVAILABLE = False
    ACTION_ROUTER_AVAILABLE = False
    SemanticRetrievalEvaluator = None
    WebSearchAugmenter = None
    CompleteActionRouter = None


# Global instances (initialized on first use)
_vector_store: Optional[PersistentVectorStore] = None
_knowledge_base: Optional[object] = None  # Agno Knowledge base instance
_semantic_evaluator: Optional[SemanticRetrievalEvaluator] = None
_web_searcher: Optional[WebSearchAugmenter] = None
_action_router: Optional[CompleteActionRouter] = None


def get_vector_store(db_path: str = "rag_database.db") -> PersistentVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = PersistentVectorStore(db_path=db_path)
    return _vector_store


def set_knowledge_base(knowledge_base: object) -> None:
    """Set the global Agno Knowledge base instance."""
    global _knowledge_base
    _knowledge_base = knowledge_base


def get_knowledge_base() -> Optional[object]:
    """Get the global Agno Knowledge base instance."""
    return _knowledge_base


def get_semantic_evaluator(
    model_path: Optional[str] = None,
    enable: bool = True,
    batch_size: int = 4,
) -> Optional[SemanticRetrievalEvaluator]:
    """
    Get or create the global semantic evaluator instance.
    
    Args:
        model_path: Path to T5 model. If None, uses default.
        enable: Whether to enable semantic evaluator (requires T5 model).
        batch_size: Batch size for evaluation (default: 4, optimized for RTX 4070).
        
    Returns:
        SemanticRetrievalEvaluator instance or None if not available.
    """
    global _semantic_evaluator
    
    if not enable or not SEMANTIC_EVALUATOR_AVAILABLE:
        return None
    
    if _semantic_evaluator is None:
        try:
            # Use default model path if not specified
            if model_path is None:
                model_path = os.getenv("T5_EVALUATOR_PATH", "finetuned_t5_evaluator")
            
            # Get batch size from environment or use default
            batch_size = int(os.getenv("T5_BATCH_SIZE", batch_size))
            
            print(f"🚀 初始化语义评估器 (批处理大小: {batch_size})...")
            _semantic_evaluator = SemanticRetrievalEvaluator(
                model_path=model_path,
                batch_size=batch_size,
            )
        except Exception as e:
            # Fallback to lexical scoring if T5 not available
            print(f"Warning: Semantic evaluator not available: {e}. Using lexical scoring.")
            return None
    
    return _semantic_evaluator


def get_web_searcher(
    enable: bool = True,
    api_key: Optional[str] = None,
) -> Optional[WebSearchAugmenter]:
    """
    Get or create the global web searcher instance.
    
    Args:
        enable: Whether to enable web search.
        api_key: Search API key. If None, uses environment variable.
        
    Returns:
        WebSearchAugmenter instance or None if not available.
    """
    global _web_searcher
    
    if not enable or not WEB_SEARCH_AVAILABLE:
        return None
    
    if _web_searcher is None:
        try:
            _web_searcher = WebSearchAugmenter(api_key=api_key)
        except Exception as e:
            print(f"Warning: Web search not available: {e}. External search disabled.")
            return None
    
    return _web_searcher


def get_action_router(
    use_semantic: bool = True,
    use_web_search: bool = True,
) -> Optional[CompleteActionRouter]:
    """
    Get or create the global action router instance.
    
    Args:
        use_semantic: Whether to use semantic evaluator.
        use_web_search: Whether to use web search.
        
    Returns:
        CompleteActionRouter instance or None if not available.
    """
    global _action_router
    
    if not ACTION_ROUTER_AVAILABLE:
        return None
    
    if _action_router is None:
        evaluator = get_semantic_evaluator(enable=use_semantic) if use_semantic else None
        web_searcher = get_web_searcher(enable=use_web_search) if use_web_search else None
        
        _action_router = CompleteActionRouter(
            evaluator=evaluator,
            web_searcher=web_searcher,
        )
    
    return _action_router


def _upload_pdf_document_impl(
    file_path: str,
    doc_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Upload a PDF document to the knowledge base.
    
    Args:
        file_path: Path to the PDF file (absolute or relative to current directory).
        doc_id: Optional custom document ID. If not provided, uses the file stem.
    
    Returns:
        Dictionary with doc_id and passage_count.
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }
    
    if not pdf_path.suffix.lower() == ".pdf":
        return {
            "success": False,
            "error": f"File is not a PDF: {file_path}",
        }
    
    try:
        # Process PDF
        processor = DocumentProcessor()
        if doc_id is None:
            doc_id = pdf_path.stem
        
        doc_id_processed, passages = processor.process_pdf(str(pdf_path))
        
        # Check if we got any passages
        if not passages:
            return {
                "success": False,
                "error": "No passages extracted from PDF. The PDF may be empty, corrupted, or contain only images.",
                "doc_id": doc_id,
                "page_count": 0,
            }
        
        # Update doc_id in all passages to match the specified doc_id
        for passage in passages:
            passage.doc_id = doc_id
        
        # Store in Agno Knowledge database
        knowledge_base = get_knowledge_base()
        
        if knowledge_base is None:
            return {
                "success": False,
                "error": "Knowledge base is not configured. Please ensure Agno Knowledge database is properly initialized.",
                "doc_id": doc_id,
            }
        
        # Combine passages into document text
        doc_text = "\n\n".join([
            f"[Page {p.page if p.page else 'N/A'}]\n{p.text}"
            for p in passages
        ])
        
        # Add content to Agno Knowledge using text_content
        # This ensures we use the already-processed passages
        knowledge_base.add_content(
            text_content=doc_text,
            metadata={
                "doc_id": doc_id,
                "file_path": str(pdf_path.absolute()),
                "original_doc_id": doc_id_processed,
                "file_size": pdf_path.stat().st_size,
            },
        )
        
        log_to_console(f"[成功] 已上传PDF文档: {doc_id} ({len(passages)} 个段落)")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "passage_count": len(passages),
            "file_path": str(pdf_path.absolute()),
            "pages_with_content": len([p for p in passages if p.text.strip()]),
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": error_details,
        }


# Create tool version for Agent
@tool(
    name="upload_pdf_document",
    description=(
        "Upload a PDF document to the knowledge base. The document will be "
        "parsed into passages and stored for later retrieval. Returns the "
        "document ID and number of passages created."
    ),
)
def upload_pdf_document(
    file_path: str,
    doc_id: Optional[str] = None,
) -> Dict[str, object]:
    """Tool wrapper for Agent."""
    return _upload_pdf_document_impl(file_path, doc_id)


def _upload_pdf_directory_impl(
    directory: str,
    pattern: str = "*.pdf",
    recursive: bool = False,
) -> Dict[str, object]:
    """
    Upload all PDF files in a directory to the knowledge base.
    
    Args:
        directory: Directory path containing PDF files
        pattern: File pattern to match (default: "*.pdf")
        recursive: If True, search subdirectories recursively
    
    Returns:
        Dictionary with upload results and statistics
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return {
            "success": False,
            "error": f"Directory not found: {directory}",
        }
    
    if not dir_path.is_dir():
        return {
            "success": False,
            "error": f"Not a directory: {directory}",
        }
    
    # Find PDF files
    if recursive:
        pdf_files = list(dir_path.rglob(pattern))
    else:
        pdf_files = list(dir_path.glob(pattern))
    
    if not pdf_files:
        return {
            "success": False,
            "error": f"No PDF files found in {directory}",
            "files_found": 0,
        }
    
    log_to_console(f"\n批量上传开始: 找到 {len(pdf_files)} 个PDF文件")
    if recursive:
        log_to_console(f"   (包含子目录)")
    log_to_console("-" * 60)
    
    # Upload statistics
    results = []
    success_count = 0
    fail_count = 0
    total_passages = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        log_to_console(f"\n[{i}/{len(pdf_files)}] 处理: {pdf_file.name}")
        result = _upload_pdf_document_impl(file_path=str(pdf_file))
        
        file_result = {
            "file": str(pdf_file),
            "file_name": pdf_file.name,
            "success": result.get("success", False),
        }
        
        if result.get("success"):
            success_count += 1
            total_passages += result.get("passage_count", 0)
            file_result["doc_id"] = result.get("doc_id")
            file_result["passage_count"] = result.get("passage_count", 0)
            log_to_console(f"  [成功] {result.get('passage_count', 0)} 个段落")
        else:
            fail_count += 1
            file_result["error"] = result.get("error", "Unknown error")
            log_to_console(f"  [失败] {result.get('error', 'Unknown error')}")
        
        results.append(file_result)
    
    # Summary
    log_to_console("\n" + "=" * 60)
    log_to_console("批量上传完成:")
    log_to_console(f"  [成功] {success_count}/{len(pdf_files)}")
    log_to_console(f"  [失败] {fail_count}/{len(pdf_files)}")
    log_to_console(f"  总段落数: {total_passages}")
    log_to_console("=" * 60)
    
    return {
        "success": True,
        "total_files": len(pdf_files),
        "success_count": success_count,
        "fail_count": fail_count,
        "total_passages": total_passages,
        "results": results,
    }


@tool(
    name="upload_pdf_directory",
    description=(
        "Upload all PDF files from a directory to the knowledge base. "
        "Searches for PDF files matching the pattern and uploads them one by one. "
        "Returns statistics about successful and failed uploads."
    ),
)
def upload_pdf_directory(
    directory: str,
    pattern: str = "*.pdf",
    recursive: bool = False,
) -> Dict[str, object]:
    """
    Tool wrapper for Agent to upload multiple PDFs from a directory.
    
    Args:
        directory: Directory path containing PDF files
        pattern: File pattern to match (default: "*.pdf")
        recursive: If True, search subdirectories recursively (default: False)
    
    Returns:
        Dictionary with upload statistics and per-file results
    """
    return _upload_pdf_directory_impl(directory, pattern, recursive)


def _extract_doc_id_pattern(query: str) -> Optional[str]:
    """
    Extract potential document ID pattern from query (e.g., "GB146", "GB19402-2012", "GB 12352—2018").
    Looks for patterns like: GB followed by numbers, or alphanumeric codes.
    Improved to handle standard numbers with years and Chinese dashes (—).
    """
    import re
    # Normalize query: replace Chinese dash (—) with regular dash (-) for easier matching
    normalized_query = query.replace('—', '-').replace('–', '-')
    
    # Pattern: GB followed by numbers, or alphanumeric codes at start of query
    patterns = [
        r'\b(GB\s*\d+[.\-\d]*)',  # GB followed by numbers (with optional spaces, dots, dashes)
        r'\b([A-Z]{2,}\s*\d+[.\-\d]*)',  # 2+ letters followed by numbers
        r'GB\s*(\d+)',  # GB followed by numbers (extract just the number part)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, normalized_query.upper())
        if match:
            # Extract the matched group
            matched = match.group(1) if match.lastindex else match.group(0)
            # Remove spaces and extract base pattern
            matched = matched.replace(' ', '').replace('—', '-').replace('–', '-')
            # Extract just the base pattern (e.g., "GB12352" from "GB12352-2018" or "GB 12352—2018")
            if matched.startswith('GB'):
                # Extract number part, handling various separators
                number_part = matched[2:]
                # Split by common separators: dash, dot, underscore
                number_part = number_part.split('-')[0].split('.')[0].split('_')[0]
                doc_id_base = 'GB' + number_part
            else:
                doc_id_base = matched.split('-')[0].split('.')[0].split('_')[0]
            return doc_id_base
    
    return None


def _query_documents_impl(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.15,
    decompose_mode: str = "excerption",
    doc_id_filter: Optional[str] = None,
) -> Dict[str, object]:
    """
    Query documents using CRAG retrieval.
    
    Args:
        query: The question or query to search for.
        top_k: Number of top passages to return (default: from CRAG_TOP_K env var or 5).
        similarity_threshold: Minimum similarity score (default: from CRAG_SIMILARITY_THRESHOLD env var or 0.15).
        decompose_mode: CRAG decompose mode: 'fixed_num', 'excerption', or 'selection' 
            (default: from CRAG_DECOMPOSE_MODE env var or 'excerption').
        doc_id_filter: Optional document ID or prefix to filter results.
    
    Returns:
        Dictionary with retrieved passages, CRAG action, and context.
    """
    # CRITICAL: Import os FIRST before any other operations
    import os
    import time
    import sqlite3
    total_start = time.time()
    
    # Load CRAG thresholds from environment variables if available
    crag_upper_threshold = float(os.getenv("CRAG_UPPER_THRESHOLD", "0.6"))
    crag_lower_threshold = float(os.getenv("CRAG_LOWER_THRESHOLD", "0.2"))
    
    # Ensure query is a proper Unicode string
    if not isinstance(query, str):
        try:
            query = str(query, 'utf-8', errors='ignore')
        except Exception:
            query = str(query)
    
    # Direct output using unified logging function
    log_to_console("")
    log_to_console("="*60)
    log_to_console("开始检索查询")
    log_to_console("="*60)
    log_to_console(f"查询: {query}")
    log_to_console(f"参数: top_k={top_k}, threshold={similarity_threshold}, mode={decompose_mode}")
    
    # Log encoding info for debugging
    try:
        import re
        has_chinese_chars = bool(re.search(r'[\u4e00-\u9fff]', query))
        query_encoding_info = f"查询编码: {type(query).__name__}, 长度: {len(query)}, 包含中文: {has_chinese_chars}"
        log_to_console(query_encoding_info)
    except Exception:
        pass
    
    try:
        # Step 1: Get Agno Knowledge base (required, no fallback)
        step_start = time.time()
        knowledge_base = get_knowledge_base()
        
        if knowledge_base is None:
            error_msg = "错误: Agno Knowledge 数据库未配置，无法进行检索"
            log_to_console(error_msg)
            return {
                "success": False,
                "action": "ambiguous",
                "message": "Knowledge base is not configured. Please ensure Agno Knowledge database is properly initialized.",
                "context": "",
                "passages": [],
                "debug": {
                    "query": query,
                    "error": "Knowledge base not available"
                },
            }
        
        log_to_console("使用 Agno Knowledge 数据库进行检索")
        # Check knowledge base stats
        try:
            content_result = knowledge_base.get_content()
            # get_content() may return (list, count) tuple or just a list
            if isinstance(content_result, tuple) and len(content_result) == 2:
                content_list, total_count = content_result
            elif isinstance(content_result, list):
                content_list = content_result
                total_count = len(content_list)
            else:
                content_list = []
                total_count = 0
            stats_msg = f"知识库统计: {total_count} 项内容"
            log_to_console(stats_msg)
        except Exception as e:
            log_to_console(f"无法获取知识库统计: {str(e)}")
        
        init_msg = f"初始化知识库: {time.time() - step_start:.3f}秒"
        log_to_console(init_msg)
        
        # Try to detect document ID pattern in query if not explicitly provided
        step_start = time.time()
        if not doc_id_filter:
            doc_id_filter = _extract_doc_id_pattern(query)
        if doc_id_filter:
            filter_msg = f"检测到文档ID过滤: {doc_id_filter}"
            log_to_console(filter_msg)
        
        doc_id_msg = f"文档ID检测: {time.time() - step_start:.3f}秒"
        log_to_console(doc_id_msg)
        
        # Step 2: Vector search using Agno Knowledge (only)
        search_start = time.time()
        all_hits = []
        search_calls = 0
        
        def search_knowledge_base(search_query: str, max_results: int = top_k, filters: Optional[Dict] = None) -> List:
            """
            Search function using Agno Knowledge database only.
            Returns list of (score, Passage) tuples.
            """
            try:
                # Use Agno Knowledge search (synchronous)
                results = knowledge_base.search(
                    query=search_query,
                    max_results=max_results,
                    filters=filters if filters else {}
                )
                
                # Ensure results is a list
                if results is None:
                    log_to_console(f"  [调试] 搜索结果为 None")
                    results = []
                if not isinstance(results, list):
                    results = list(results) if hasattr(results, '__iter__') else []
                
                log_to_console(f"  [调试] 搜索返回 {len(results)} 个原始结果")
                
                # Convert Agno Knowledge results to (score, Passage) format
                # Agno Knowledge.search() returns List[Document] where Document has:
                # - content: str (the text content)
                # - meta_data: Dict[str, Any] (metadata, note: meta_data not metadata)
                # - id: Optional[str] (document ID)
                # - name: Optional[str] (document name)
                # - reranking_score: Optional[float] (similarity score)
                from dataclasses import dataclass
                @dataclass
                class SimplePassage:
                    doc_id: str = ""
                    page: Optional[int] = None
                    text: str = ""
                
                hits = []
                if results:
                    for idx, result in enumerate(results):
                        try:
                            # Agno Knowledge returns Document objects
                            # Extract content directly from Document.content
                            text = ""
                            if hasattr(result, 'content'):
                                text = str(result.content) if result.content else ""
                            elif isinstance(result, dict) and 'content' in result:
                                text = str(result['content']) if result['content'] else ""
                            
                            # Skip empty text
                            if not text or not text.strip():
                                if idx < 3:
                                    log_to_console(f"  [调试] 结果 {idx+1} 文本为空，跳过")
                                continue
                            
                            # Extract score - use reranking_score if available, otherwise try to get from similarity
                            # Note: Agno Knowledge search may return similarity scores in different formats
                            score = 1.0  # Default score for semantic search results
                            if hasattr(result, 'reranking_score') and result.reranking_score is not None:
                                score = float(result.reranking_score)
                            elif hasattr(result, 'similarity') and result.similarity is not None:
                                score = float(result.similarity)
                            elif isinstance(result, dict):
                                if 'reranking_score' in result and result['reranking_score'] is not None:
                                    score = float(result['reranking_score'])
                                elif 'similarity' in result and result['similarity'] is not None:
                                    score = float(result['similarity'])
                            
                            # Normalize score: Agno Knowledge semantic search typically returns scores in [0, 1] range
                            # If score seems too high (e.g., > 1.0), it might be a distance metric, so we normalize it
                            if score > 1.0:
                                # Likely a distance metric, convert to similarity (inverse)
                                score = 1.0 / (1.0 + score)
                            elif score < 0:
                                # Negative scores might indicate low relevance, normalize to [0, 1]
                                score = max(0.0, min(1.0, (score + 1.0) / 2.0))
                            
                            # Extract doc_id from meta_data (note: meta_data not metadata)
                            doc_id = ""
                            page = None
                            
                            # Try meta_data first (this is the correct attribute name for Document)
                            if hasattr(result, 'meta_data') and result.meta_data:
                                meta_data = result.meta_data
                                if isinstance(meta_data, dict):
                                    doc_id = str(meta_data.get('doc_id', '')) or str(meta_data.get('document_id', ''))
                                    page = meta_data.get('page') or meta_data.get('page_number')
                            
                            # Also try metadata (some versions might use this)
                            if not doc_id:
                                if hasattr(result, 'metadata') and result.metadata:
                                    metadata = result.metadata
                                    if isinstance(metadata, dict):
                                        doc_id = str(metadata.get('doc_id', '')) or str(metadata.get('document_id', ''))
                                        page = metadata.get('page') or metadata.get('page_number')
                            
                            # If no doc_id from metadata, try content_id or id
                            if not doc_id:
                                if hasattr(result, 'content_id') and result.content_id:
                                    doc_id = str(result.content_id)
                                elif hasattr(result, 'id') and result.id:
                                    doc_id = str(result.id)
                                elif hasattr(result, 'name') and result.name:
                                    # Use name as fallback
                                    doc_id = str(result.name)
                            
                            # If still no doc_id, try from dict format
                            if not doc_id and isinstance(result, dict):
                                doc_id = str(result.get('content_id', '')) or str(result.get('id', '')) or str(result.get('name', ''))
                            
                            # Log first few results for debugging
                            if idx < 3:
                                text_preview = text[:100].replace('\n', ' ') if text else "无文本"
                                log_to_console(f"  [调试] 结果 {idx+1}: 分数={score:.4f}, doc_id={doc_id}, 文本长度={len(text)}, 预览={text_preview}...")
                            
                            passage = SimplePassage(doc_id=doc_id, page=page, text=text)
                            hits.append((score, passage))
                        except Exception as e:
                            # Skip problematic results but log them
                            log_to_console(f"  [警告] 跳过无效搜索结果 {idx+1}: {str(e)}")
                            import traceback
                            if idx < 3:  # Only log traceback for first few errors
                                log_to_console(f"  错误详情: {traceback.format_exc()}")
                            continue
                
                log_to_console(f"  [调试] 成功提取 {len(hits)} 个有效结果")
                return hits
            except Exception as e:
                log_to_console(f"  [错误] Agno Knowledge 搜索失败: {str(e)}")
                import traceback
                log_to_console(f"  错误详情: {traceback.format_exc()}")
                # Return empty list instead of falling back to old database
                return []
        
        # Strategy 1: Original query (with optional doc_id filter)
        strategy1_msg = f"\n策略1: 原始查询检索"
        log_to_console(strategy1_msg)
        
        # First, try search without filters to get all relevant results
        # Agno Knowledge uses semantic search, so it should find relevant content even without exact doc_id match
        call_start = time.time()
        hits = search_knowledge_base(query, max_results=top_k * 5, filters=None)
        call_time = time.time() - call_start
        search_calls += 1
        threshold_msg = f"  查询（无过滤）: 找到 {len(hits)} 个结果 ({call_time:.3f}秒)"
        log_to_console(threshold_msg)
        
        if hits:
            all_hits.extend(hits)
            
            # If doc_id_filter is specified, also try filtering and prioritize matching results
            if doc_id_filter:
                log_to_console(f"  尝试使用文档ID过滤: {doc_id_filter}")
                filters = {"doc_id": doc_id_filter}
                call_start = time.time()
                filtered_hits = search_knowledge_base(query, max_results=top_k * 3, filters=filters)
                call_time = time.time() - call_start
                search_calls += 1
                filter_msg = f"  查询（ID过滤）: 找到 {len(filtered_hits)} 个结果 ({call_time:.3f}秒)"
                log_to_console(filter_msg)
                
                if filtered_hits:
                    # Prioritize filtered results by prepending them
                    all_hits = filtered_hits + all_hits
                    found_msg = f"  [完成] 文档ID过滤找到结果，优先显示"
                    log_to_console(found_msg)
                else:
                    # If filter didn't find results, check if any results match the doc_id pattern
                    matching_hits = []
                    for score, passage in hits:
                        if doc_id_filter.lower() in passage.doc_id.lower() or passage.doc_id.lower().startswith(doc_id_filter.lower()):
                            matching_hits.append((score, passage))
                    if matching_hits:
                        log_to_console(f"  在搜索结果中发现 {len(matching_hits)} 个匹配文档ID的结果")
                        # Prioritize matching results
                        all_hits = matching_hits + all_hits
        
        # Strategy 2: Extract keywords and search with them (optional)
        # Agno Knowledge uses semantic search, but keyword-based search can still help
        # Only do this if we haven't found enough results yet
        if len(all_hits) < top_k:
            step_start = time.time()
            from persistent_vector_store import _extract_keywords
            keywords = _extract_keywords(query)
            strategy2_msg = f"\n策略2: 关键词检索 (关键词: {keywords})"
            log_to_console(strategy2_msg)
            
            # For Chinese queries, search with each keyword individually
            import re
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
            
            if keywords:
                if has_chinese:
                    # Chinese: search with each keyword separately
                    sorted_keywords = sorted(keywords, key=lambda x: (len(x), -keywords.index(x)), reverse=True)
                    
                    for keyword in sorted_keywords[:5]:  # Limit to top 5 keywords
                        call_start = time.time()
                        keyword_hits = search_knowledge_base(keyword, max_results=top_k * 2, filters=None)
                        call_time = time.time() - call_start
                        search_calls += 1
                        keyword_msg = f"  关键词 '{keyword}': 找到 {len(keyword_hits)} 个结果 ({call_time:.3f}秒)"
                        log_to_console(keyword_msg)
                        
                        if keyword_hits:
                            all_hits.extend(keyword_hits)
                            if len(all_hits) >= top_k * 3:
                                break
                else:
                    # English: search with combined keywords
                    keyword_query = " ".join(keywords)
                    call_start = time.time()
                    keyword_hits = search_knowledge_base(keyword_query, max_results=top_k * 2, filters=None)
                    call_time = time.time() - call_start
                    search_calls += 1
                    threshold_msg = f"  关键词查询: 找到 {len(keyword_hits)} 个结果 ({call_time:.3f}秒)"
                    log_to_console(threshold_msg)
                    
                    if keyword_hits:
                        all_hits.extend(keyword_hits)
            
            keyword_time_msg = f"关键词提取和检索: {time.time() - step_start:.3f}秒"
            log_to_console(keyword_time_msg)
        
        # Strategy 3: If doc_id_filter is specified but no results found, try searching with just the doc_id
        if doc_id_filter and not all_hits:
            strategy3_msg = f"\n策略3: 使用文档ID直接搜索"
            log_to_console(strategy3_msg)
            # Try searching with just the doc_id as query
            call_start = time.time()
            doc_id_hits = search_knowledge_base(doc_id_filter, max_results=top_k * 3, filters=None)
            call_time = time.time() - call_start
            search_calls += 1
            doc_id_msg = f"  文档ID查询 '{doc_id_filter}': 找到 {len(doc_id_hits)} 个结果 ({call_time:.3f}秒)"
            log_to_console(doc_id_msg)
            
            if doc_id_hits:
                all_hits.extend(doc_id_hits)
        
        search_total_time = time.time() - search_start
        search_total_msg = f"\n向量检索总耗时: {search_total_time:.3f}秒 (共 {search_calls} 次搜索调用)"
        log_to_console(search_total_msg)
        if search_calls > 0:
            avg_msg = f"   平均每次搜索: {search_total_time/search_calls:.3f}秒"
            log_to_console(avg_msg)
        
        # Deduplicate and sort by score
        step_start = time.time()
        seen_passages = set()
        unique_hits = []
        for score, passage in sorted(all_hits, key=lambda x: x[0], reverse=True):
            passage_key = (passage.doc_id, passage.page, passage.text[:100])
            if passage_key not in seen_passages:
                seen_passages.add(passage_key)
                unique_hits.append((score, passage))
        
        hits = unique_hits[:top_k * 2]  # Keep top results
        dedup_msg = f"去重和排序: {time.time() - step_start:.3f}秒"
        log_to_console(dedup_msg)
        final_msg = f"最终检索结果: {len(hits)} 个段落 (去重前: {len(all_hits)})"
        log_to_console(final_msg)
        
        # Log document IDs found in results for debugging
        if hits:
            doc_ids_found = set()
            for score, passage in hits:
                if passage.doc_id:
                    doc_ids_found.add(passage.doc_id)
            if doc_ids_found:
                doc_ids_msg = f"找到的文档ID: {', '.join(sorted(doc_ids_found))}"
                log_to_console(doc_ids_msg)
            if doc_id_filter:
                matching_docs = [doc_id for doc_id in doc_ids_found if doc_id_filter.lower() in doc_id.lower() or doc_id.lower().startswith(doc_id_filter.lower())]
                if matching_docs:
                    match_msg = f"匹配文档ID过滤 '{doc_id_filter}': {', '.join(matching_docs)}"
                    log_to_console(match_msg)
                else:
                    no_match_msg = f"未找到匹配文档ID过滤 '{doc_id_filter}' 的文档（但找到了其他文档）"
                    log_to_console(no_match_msg)
        
        if not hits:
            # Try to get some debug info about why no results
            no_results_msg = f"\n[警告] 未找到任何结果，尝试诊断..."
            log_to_console(no_results_msg)
            
            # Check database stats
            try:
                # Check Agno Knowledge stats
                try:
                    content_result = knowledge_base.get_content()
                    # get_content() may return (list, count) tuple or just a list
                    if isinstance(content_result, tuple) and len(content_result) == 2:
                        content_list, total_count = content_result
                    elif isinstance(content_result, list):
                        content_list = content_result
                        total_count = len(content_list)
                    else:
                        content_list = []
                        total_count = 0
                    db_stats_msg = f"  知识库统计: {total_count} 项内容"
                    log_to_console(db_stats_msg)
                except Exception as e:
                    log_to_console(f"  无法获取知识库统计: {str(e)}")
                
                # Try a broader search
                debug_hits = search_knowledge_base(query, max_results=10, filters=None)
                if debug_hits:
                    debug_found_msg = f"  使用更宽泛的搜索找到 {len(debug_hits)} 个结果"
                    log_to_console(debug_found_msg)
                    preview_msg = f"     前3个结果:"
                    log_to_console(preview_msg)
                    for i, (score, passage) in enumerate(debug_hits[:3], 1):
                        preview = passage.text[:100].replace('\n', ' ')
                        result_msg = f"       {i}. 分数: {score:.4f}, 文档: {passage.doc_id}, 预览: {preview}..."
                        log_to_console(result_msg)
                else:
                    no_debug_msg = f"  [错误] 即使使用更宽泛的搜索也未找到任何结果"
                    log_to_console(no_debug_msg)
                    suggestion_msg = f"  建议: 检查查询词是否与文档内容匹配，或尝试使用文档ID过滤"
                    log_to_console(suggestion_msg)
            except Exception as e:
                error_msg = f"  [警告] 诊断失败: {e}"
                log_to_console(error_msg)
            
            return {
                "success": False,
                "action": "ambiguous",
                "message": "No relevant passages found in the knowledge base. The query may not match any content in the uploaded documents.",
                "context": "",
                "passages": [],
                "debug": {
                    "query": query,
                    "search_calls": search_calls,
                },
            }
        
        # Step 3: Extract candidate passages and clean encoding artifacts
        step_start = time.time()
        candidate_passages = []
        import re
        for hit in hits:
            text = hit[1].text
            # Clean PDF encoding artifacts
            if text:
                # Remove PDF font encoding artifacts like /G21, /G22, etc.
                text = re.sub(r'/[Gg][0-9A-Fa-f]{2}\s*', '', text)
                # Remove other common PDF encoding artifacts
                text = re.sub(r'\\[0-9]{3}', '', text)  # Remove octal escapes
                # Clean up multiple spaces but preserve newlines
                text = re.sub(r'[ \t]+', ' ', text)
                text = text.strip()
            if text:  # Only add non-empty cleaned text
                candidate_passages.append(text)
        
        # Limit candidate passages to reduce evaluation time
        max_candidates = min(len(candidate_passages), 10)  # Limit to 10 passages for speed
        candidate_passages = candidate_passages[:max_candidates]
        
        extract_msg = f"\n提取候选段落: {time.time() - step_start:.3f}秒"
        log_to_console(extract_msg)
        candidate_msg = f"候选段落数: {len(candidate_passages)}"
        log_to_console(candidate_msg)
        
        # Step 4: CRAG evaluation with smart optimization
        crag_start = time.time()
        use_complete_crag = os.getenv("USE_COMPLETE_CRAG", "true").lower() == "true"
        
        # Initialize action_router to None (will be set in else branch if needed)
        action_router = None
        
        # Check if fast path optimization is disabled (for performance evaluation)
        disable_fast_path = os.getenv("DISABLE_FAST_PATH", "false").lower() == "true"
        
        # Smart CRAG optimization: Skip expensive T5 evaluation when results are clearly relevant
        # Conditions for fast path (skip full CRAG):
        # 1. Found enough results (>= top_k)
        # 2. Top results have reasonable scores (>= 0.3, indicating semantic similarity)
        # 3. Query contains document ID and we found matching documents
        # 4. Results contain query keywords (simple keyword check)
        skip_full_crag = False
        fast_path_reason = ""
        
        if disable_fast_path:
            # Fast path is disabled - force complete CRAG evaluation
            log_to_console("\n[性能评估模式] 快速路径已禁用，强制使用完整 CRAG")
            skip_full_crag = False
        elif len(hits) >= top_k:
            # Check if top results have good scores
            top_scores = [score for score, _ in hits[:top_k]]
            avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
            max_score = max(top_scores) if top_scores else 0.0
            
            # Fast path condition 1: Good semantic similarity scores
            if max_score >= 0.5 or avg_top_score >= 0.4:
                skip_full_crag = True
                fast_path_reason = f"高质量检索结果（最高分={max_score:.3f}, 平均分={avg_top_score:.3f}）"
            
            # Fast path condition 2: Document ID match found
            elif doc_id_filter:
                # Check if any results match the doc_id pattern
                matching_docs = [doc_id for doc_id in doc_ids_found if doc_id_filter.lower() in doc_id.lower() or doc_id.lower().startswith(doc_id_filter.lower())]
                if matching_docs:
                    skip_full_crag = True
                    fast_path_reason = f"找到匹配文档ID '{doc_id_filter}' 的文档"
            
            # Fast path condition 3: Query keywords found in results
            else:
                # Simple keyword matching check
                query_lower = query.lower()
                query_keywords = set(query_lower.split())
                # Check if top results contain query keywords
                keyword_matches = 0
                for score, passage in hits[:min(5, len(hits))]:
                    passage_lower = passage.text.lower()
                    matched_keywords = sum(1 for kw in query_keywords if len(kw) > 2 and kw in passage_lower)
                    if matched_keywords >= min(2, len(query_keywords) // 2):
                        keyword_matches += 1
                
                if keyword_matches >= min(3, len(hits[:5])):
                    skip_full_crag = True
                    fast_path_reason = f"检索结果包含查询关键词（{keyword_matches}/{min(5, len(hits))} 个结果匹配）"
        
        # Use fast path if conditions are met
        if skip_full_crag and use_complete_crag:
            log_to_console(f"\nCRAG评估模式: 快速路径（跳过完整CRAG）")
            log_to_console(f"  原因: {fast_path_reason}")
            log_to_console(f"  直接使用检索结果，跳过T5语义评估以提升速度")
            
            # Use simple CRAG evaluation (lexical-based, fast)
            from crag_layer import crag_evaluate_and_route
            crag_result = crag_evaluate_and_route(
                query=query,
                candidate_passages=candidate_passages,
                top_k=min(len(candidate_passages), top_k),
                upper_threshold=crag_upper_threshold,
                lower_threshold=crag_lower_threshold,
                decompose_mode=decompose_mode,
            )
            
            action = crag_result.get("action", "correct")  # Default to "correct" for fast path
            global_score = crag_result.get("global_score", 0.5)  # Default to reasonable score
            selected_strips = crag_result.get("selected_strips", candidate_passages[:top_k])
            
            # Clean encoding artifacts in final output
            cleaned_strips = []
            for strip in selected_strips:
                if strip:
                    cleaned = re.sub(r'/[Gg][0-9A-Fa-f]{2}\s*', '', strip)
                    cleaned = re.sub(r'\\[0-9]{3}', '', cleaned)
                    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                    cleaned = cleaned.strip()
                    if cleaned:
                        cleaned_strips.append(cleaned)
            
            context = "\n\n".join(f"[Passage {i+1}]\n{strip}" for i, strip in enumerate(cleaned_strips))
            
            fast_result_msg = f"  [完成] 快速路径: 动作={action}, 分数={global_score:.4f}, 片段数={len(cleaned_strips)}"
            log_to_console(fast_result_msg)
        else:
            # Use full CRAG evaluation (with T5 semantic evaluator)
            action_router = get_action_router() if use_complete_crag else None
            
            crag_mode_msg = f"\nCRAG评估模式: {'完整CRAG' if action_router else '基础CRAG'}"
            log_to_console(crag_mode_msg)
            
            if action_router:
                # Use complete CRAG with action routing
                router_msg = "  使用完整动作路由..."
                log_to_console(router_msg)
                router_result = action_router.route_action(
                    query=query,
                    retrieved_docs=candidate_passages,
                    decompose_mode=decompose_mode,
                )
                
                action = router_result.get("action", "ambiguous")
                knowledge = router_result.get("knowledge", [])
                global_score = router_result.get("global_score", 0.0)
                
                action_msg = f"  [完成] 动作: {action}, 全局分数: {global_score:.4f}, 知识片段数: {len(knowledge)}"
                log_to_console(action_msg)
                
                # Format context from knowledge (clean encoding artifacts in final output)
                cleaned_knowledge = []
                for strip in knowledge:
                    if strip:
                        # Clean PDF encoding artifacts in final output
                        cleaned = re.sub(r'/[Gg][0-9A-Fa-f]{2}\s*', '', strip)
                        cleaned = re.sub(r'\\[0-9]{3}', '', cleaned)
                        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                        cleaned = cleaned.strip()
                        if cleaned:
                            cleaned_knowledge.append(cleaned)
                
                context = "\n\n".join(f"[Knowledge {i+1}]\n{strip}" for i, strip in enumerate(cleaned_knowledge))
                
                # Include external knowledge info if available
                external_knowledge = router_result.get("external_knowledge", [])
                if external_knowledge:
                    cleaned_external = []
                    for strip in external_knowledge:
                        if strip:
                            cleaned = re.sub(r'/[Gg][0-9A-Fa-f]{2}\s*', '', strip)
                            cleaned = re.sub(r'\\[0-9]{3}', '', cleaned)
                            cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                            cleaned = cleaned.strip()
                            if cleaned:
                                cleaned_external.append(cleaned)
                    if cleaned_external:
                        external_msg = f"  外部知识: {len(cleaned_external)} 个片段"
                        log_to_console(external_msg)
                        context += "\n\n[External Knowledge]\n"
                        context += "\n\n".join(f"[External {i+1}]\n{strip}" for i, strip in enumerate(cleaned_external))
            else:
                # Fall back to basic CRAG evaluation
                basic_msg = "  使用基础CRAG评估..."
                log_to_console(basic_msg)
                crag_result = crag_evaluate_and_route(
                    query=query,
                    candidate_passages=candidate_passages,
                    top_k=min(len(candidate_passages), top_k),
                    upper_threshold=crag_upper_threshold,
                    lower_threshold=crag_lower_threshold,
                    decompose_mode=decompose_mode,
                )
                
                action = crag_result.get("action", "ambiguous")
                global_score = crag_result.get("global_score", 0.0)
                selected_strips = crag_result.get("selected_strips", [])
                basic_result_msg = f"  [完成] 动作: {action}, 全局分数: {global_score:.4f}, 选择片段数: {len(selected_strips)}"
                log_to_console(basic_result_msg)
                
                # Clean encoding artifacts in final output
                cleaned_strips = []
                for strip in selected_strips:
                    if strip:
                        cleaned = re.sub(r'/[Gg][0-9A-Fa-f]{2}\s*', '', strip)
                        cleaned = re.sub(r'\\[0-9]{3}', '', cleaned)
                        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                        cleaned = cleaned.strip()
                        if cleaned:
                            cleaned_strips.append(cleaned)
                
                context = "\n\n".join(f"[Passage {i+1}]\n{strip}" for i, strip in enumerate(cleaned_strips))
        
        crag_time = time.time() - crag_start
        crag_time_msg = f"CRAG评估总耗时: {crag_time:.3f}秒"
        log_to_console(crag_time_msg)
        
        # Step 5: Format results
        step_start = time.time()
        passages_info = []
        for score, passage in hits[:top_k]:
            try:
                # Ensure all text is properly encoded
                preview_text = passage.text[:200] + "..." if len(passage.text) > 200 else passage.text
                if not isinstance(preview_text, str):
                    preview_text = str(preview_text, 'utf-8', errors='ignore')
                
                passages_info.append({
                    "score": round(score, 4),
                    "doc_id": str(passage.doc_id) if passage.doc_id else "",
                    "page": passage.page,
                    "preview": preview_text,
                })
            except Exception:
                # Skip problematic passages
                continue
        
        # Group results by document ID
        doc_id_counts = {}
        for score, passage in hits:
            try:
                doc_id = str(passage.doc_id) if passage.doc_id else ""
                doc_id_counts[doc_id] = doc_id_counts.get(doc_id, 0) + 1
            except Exception:
                continue
        
        format_time = time.time() - step_start
        total_time = time.time() - total_start
        
        format_msg = f"\n结果格式化: {format_time:.3f}秒"
        log_to_console(format_msg)
        
        summary_sep = f"\n{'='*60}"
        log_to_console(summary_sep)
        
        total_msg = f"[完成] 检索完成 - 总耗时: {total_time:.2f}秒"
        log_to_console(total_msg)
        
        search_msg = f"   检索: {search_total_time:.2f}秒 ({search_total_time/total_time*100:.1f}%)"
        log_to_console(search_msg)
        
        crag_percent_msg = f"   CRAG评估: {crag_time:.2f}秒 ({crag_time/total_time*100:.1f}%)"
        log_to_console(crag_percent_msg)
        
        other_msg = f"   其他: {total_time - search_total_time - crag_time:.2f}秒"
        log_to_console(other_msg)
        
        end_sep = f"{'='*60}"
        log_to_console(end_sep)
        
        return {
            "success": True,
            "action": action,
            "global_score": global_score,
            "context": context,
            "passages": passages_info,
            "num_retrieved": len(hits),
            "num_selected": len(context.split("\n\n")) if context else 0,
            "doc_id_filter": doc_id_filter,
            "documents_found": list(doc_id_counts.keys()),
            "doc_id_counts": doc_id_counts,
            "using_complete_crag": action_router is not None,
        }
    except Exception as e:
        # Ensure os is available for error reporting
        import os
        import traceback
        error_msg = str(e)
        # Provide more detailed error information
        error_type = type(e).__name__
        return {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "traceback": traceback.format_exc() if "os" in error_msg.lower() or "name" in error_msg.lower() else None,
        }


# Create tool version for Agent
@tool(
    name="query_documents",
    description=(
        "Query the knowledge base using CRAG (Corrective Retrieval Augmented Generation). "
        "Searches uploaded documents for relevant passages, evaluates their quality, "
        "and returns the most relevant context for answering the question. "
        "If the query contains a document ID pattern (e.g., 'GB146'), it will automatically "
        "filter results to that document. Use this tool when you need to find information from uploaded PDF documents."
    ),
)
def query_documents(
    query: str,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    decompose_mode: Optional[str] = None,
    doc_id_filter: Optional[str] = None,
) -> Dict[str, object]:
    """
    Tool wrapper for Agent.
    
    If parameters are not provided, uses values from environment variables:
    - CRAG_TOP_K (default: 5)
    - CRAG_SIMILARITY_THRESHOLD (default: 0.15)
    - CRAG_DECOMPOSE_MODE (default: 'excerption')
    """
    try:
        import os  # Ensure os is available in function scope (CRITICAL: must be first)
        # Load defaults from environment variables if not provided
        if top_k is None:
            top_k = int(os.getenv("CRAG_TOP_K", "5"))
        if similarity_threshold is None:
            similarity_threshold = float(os.getenv("CRAG_SIMILARITY_THRESHOLD", "0.15"))
        if decompose_mode is None:
            decompose_mode = os.getenv("CRAG_DECOMPOSE_MODE", "excerption")
        
        return _query_documents_impl(query, top_k, similarity_threshold, decompose_mode, doc_id_filter)
    except NameError as e:
        # Specifically handle NameError for undefined variables
        if "'os'" in str(e) or "os" in str(e).lower():
            # This should never happen if import os is at the top, but handle it anyway
            import os
            import traceback
            return {
                "success": False,
                "error": f"Import error: {str(e)}. Traceback: {traceback.format_exc()}",
                "error_type": "NameError",
                "hint": "os module should be imported at function start",
            }
        else:
            raise
    except Exception as e:
        # Catch any other exceptions
        import traceback
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


def _list_documents_impl() -> Dict[str, object]:
    """List all uploaded documents from Agno Knowledge."""
    try:
        knowledge_base = get_knowledge_base()
        
        if knowledge_base is None:
            return {
                "success": False,
                "error": "Knowledge base is not configured",
                "documents": [],
                "count": 0,
            }
        
        # Get content from Agno Knowledge
        content_result = knowledge_base.get_content()
        
        # Handle different return formats
        if isinstance(content_result, tuple) and len(content_result) == 2:
            content_list, total_count = content_result
        elif isinstance(content_result, list):
            content_list = content_result
            total_count = len(content_list)
        else:
            content_list = []
            total_count = 0
        
        # Convert Agno Knowledge content to document format
        documents = []
        seen_doc_ids = set()
        
        for content in content_list:
            try:
                # Extract doc_id from metadata
                doc_id = ""
                file_path = ""
                
                if hasattr(content, 'metadata') and content.metadata:
                    metadata = content.metadata
                    if isinstance(metadata, dict):
                        doc_id = str(metadata.get('doc_id', ''))
                        file_path = str(metadata.get('file_path', ''))
                    elif hasattr(metadata, 'doc_id'):
                        doc_id = str(metadata.doc_id)
                        file_path = str(getattr(metadata, 'file_path', ''))
                
                # If no doc_id in metadata, try to get from content attributes
                if not doc_id:
                    if hasattr(content, 'doc_id'):
                        doc_id = str(content.doc_id)
                    elif hasattr(content, 'id'):
                        # Use content ID as fallback
                        doc_id = str(content.id)
                
                # Skip if already seen (deduplicate by doc_id)
                if doc_id and doc_id in seen_doc_ids:
                    continue
                
                if doc_id:
                    seen_doc_ids.add(doc_id)
                
                # Get content name or use doc_id
                content_name = ""
                if hasattr(content, 'name'):
                    content_name = str(content.name)
                elif hasattr(content, 'id'):
                    content_name = str(content.id)
                else:
                    content_name = doc_id if doc_id else "Unknown"
                
                # Get upload time if available
                upload_time = None
                if hasattr(content, 'created_at'):
                    upload_time = str(content.created_at)
                elif hasattr(content, 'timestamp'):
                    upload_time = str(content.timestamp)
                
                documents.append({
                    "doc_id": doc_id if doc_id else content_name,
                    "file_path": file_path,
                    "name": content_name,
                    "upload_time": upload_time,
                })
            except Exception as e:
                # Skip problematic content items
                log_to_console(f"  [警告] 处理内容项时出错: {str(e)}")
                continue
        
        return {
            "success": True,
            "documents": documents,
            "count": len(documents),
            "total_content_items": total_count,
        }
    except Exception as e:
        log_to_console(f"[错误] 列出文档时出错: {str(e)}")
        import traceback
        log_to_console(f"[错误详情] {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "documents": [],
            "count": 0,
        }


# Create tool version for Agent
@tool(
    name="list_documents",
    description=(
        "List all documents in the knowledge base. Returns document IDs, "
        "file paths, and upload timestamps."
    ),
)
def list_documents() -> Dict[str, object]:
    """Tool wrapper for Agent."""
    return _list_documents_impl()


def _clear_knowledge_base_impl() -> Dict[str, object]:
    """
    Clear all documents from the Agno Knowledge base.
    This also clears the vector database to fix dimension mismatch issues.
    
    Returns:
        Success status and deletion counts.
    """
    try:
        knowledge_base = get_knowledge_base()
        
        if knowledge_base is None:
            return {
                "success": False,
                "error": "Knowledge base is not configured",
                "message": "Knowledge base is not configured",
            }
        
        # Get content count before deletion
        try:
            content_result = knowledge_base.get_content()
            if isinstance(content_result, tuple) and len(content_result) == 2:
                content_list, total_count = content_result
            elif isinstance(content_result, list):
                total_count = len(content_result)
            else:
                total_count = 0
        except Exception:
            total_count = 0
        
        # Remove all content from Agno Knowledge
        knowledge_base.remove_all_content()
        
        # Also clear the vector database to fix dimension mismatch
        # This is important when switching embedders (e.g., from OpenAI 1536-dim to SentenceTransformer 384-dim)
        try:
            if hasattr(knowledge_base, 'vector_db') and knowledge_base.vector_db:
                vector_db = knowledge_base.vector_db
                
                # Try to delete the vector database table/dataset
                if hasattr(vector_db, 'delete') or hasattr(vector_db, 'drop'):
                    try:
                        if hasattr(vector_db, 'delete'):
                            vector_db.delete()
                        elif hasattr(vector_db, 'drop'):
                            vector_db.drop()
                        log_to_console("[成功] 已清空向量数据库")
                    except Exception as e:
                        log_to_console(f"[警告] 清空向量数据库时出错（可能已为空）: {str(e)}")
                
                # Also try to delete the LanceDB directory if it exists
                if hasattr(vector_db, 'uri') or hasattr(vector_db, 'path'):
                    try:
                        import shutil
                        from pathlib import Path
                        db_path = Path(vector_db.uri if hasattr(vector_db, 'uri') else vector_db.path)
                        if db_path.exists():
                            # Delete the entire LanceDB directory
                            shutil.rmtree(db_path, ignore_errors=True)
                            log_to_console(f"[成功] 已删除向量数据库目录: {db_path}")
                    except Exception as e:
                        log_to_console(f"[警告] 删除向量数据库目录时出错: {str(e)}")
        except Exception as e:
            log_to_console(f"[警告] 清空向量数据库时出错: {str(e)}")
            # Continue anyway, as remove_all_content() should have cleared the content
        
        log_to_console(f"[成功] 已清空知识库，删除了 {total_count} 项内容")
        log_to_console("[重要] 请重新上传所有文档以使用新的嵌入器（384维 SentenceTransformer）")
        
        return {
            "success": True,
            "message": f"Knowledge base cleared successfully. Please re-upload all documents to use the new embedder (384-dim SentenceTransformer).",
            "documents_deleted": total_count,
            "passages_deleted": total_count,  # In Agno Knowledge, each content item may contain multiple chunks
        }
    except Exception as e:
        log_to_console(f"[错误] 清空知识库时出错: {str(e)}")
        import traceback
        log_to_console(f"[错误详情] {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
        }


def _delete_document_impl(doc_id: str) -> Dict[str, object]:
    """
    Delete a document from the Agno Knowledge base by doc_id.
    
    Args:
        doc_id: The document ID to delete (matches metadata['doc_id']).
    
    Returns:
        Success status and message.
    """
    try:
        knowledge_base = get_knowledge_base()
        
        if knowledge_base is None:
            return {
                "success": False,
                "error": "Knowledge base is not configured",
                "message": f"Document '{doc_id}' cannot be deleted: Knowledge base is not configured",
            }
        
        # Get all content to find items matching the doc_id
        try:
            content_result = knowledge_base.get_content()
            if isinstance(content_result, tuple) and len(content_result) == 2:
                content_list, total_count = content_result
            elif isinstance(content_result, list):
                content_list = content_result
                total_count = len(content_list)
            else:
                content_list = []
                total_count = 0
        except Exception as e:
            log_to_console(f"[错误] 获取知识库内容时出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve content: {str(e)}",
            }
        
        # Find and delete content items matching the doc_id
        deleted_count = 0
        found_items = []
        
        for content in content_list:
            try:
                # Extract doc_id from metadata
                content_doc_id = ""
                if hasattr(content, 'metadata') and content.metadata:
                    metadata = content.metadata
                    if isinstance(metadata, dict):
                        content_doc_id = str(metadata.get('doc_id', ''))
                    elif hasattr(metadata, 'doc_id'):
                        content_doc_id = str(metadata.doc_id)
                
                # Check if this content matches the doc_id to delete
                if content_doc_id and (content_doc_id == doc_id or content_doc_id.startswith(doc_id) or doc_id in content_doc_id):
                    # Get content ID for deletion
                    content_id = ""
                    if hasattr(content, 'id'):
                        content_id = str(content.id)
                    elif hasattr(content, 'content_id'):
                        content_id = str(content.content_id)
                    
                    if content_id:
                        found_items.append((content_id, content_doc_id))
            except Exception as e:
                log_to_console(f"  [警告] 处理内容项时出错: {str(e)}")
                continue
        
        # Delete found items
        for content_id, content_doc_id in found_items:
            try:
                knowledge_base.remove_content_by_id(content_id)
                deleted_count += 1
                log_to_console(f"  [成功] 已删除内容项: {content_doc_id} (ID: {content_id})")
            except Exception as e:
                log_to_console(f"  [警告] 删除内容项 {content_id} 时出错: {str(e)}")
                continue
        
        if deleted_count > 0:
            log_to_console(f"[成功] 已删除文档 '{doc_id}'，共删除 {deleted_count} 项内容")
            return {
                "success": True,
                "message": f"Document '{doc_id}' deleted successfully ({deleted_count} content items removed).",
                "deleted_count": deleted_count,
            }
        else:
            log_to_console(f"[警告] 未找到匹配文档ID '{doc_id}' 的内容")
            return {
                "success": False,
                "message": f"Document '{doc_id}' not found in knowledge base.",
            }
    except Exception as e:
        log_to_console(f"[错误] 删除文档时出错: {str(e)}")
        import traceback
        log_to_console(f"[错误详情] {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
        }


# Create tool version for Agent
@tool(
    name="delete_document",
    description=(
        "Delete a document from the knowledge base by its document ID. "
        "This will remove all passages associated with the document."
    ),
)
def delete_document(doc_id: str) -> Dict[str, object]:
    """Tool wrapper for Agent."""
    return _delete_document_impl(doc_id)


# Create tool version for Agent
@tool(
    name="clear_knowledge_base",
    description=(
        "Clear all documents from the knowledge base. This will delete all "
        "uploaded PDFs and their passages. Use with caution - this action cannot be undone!"
    ),
)
def clear_knowledge_base() -> Dict[str, object]:
    """Tool wrapper for Agent."""
    return _clear_knowledge_base_impl()


# Create a tools class for easy registration
class RAGTools:
    """Collection of RAG tools for Agno Agent."""
    
    def __init__(self, db_path: str = "rag_database.db"):
        """Initialize RAG tools. db_path is kept for backward compatibility."""
        self.db_path = db_path
        # Note: Vector store initialization is no longer required
        # as we now use Agno Knowledge, but kept for backward compatibility
    
    @property
    def tools(self):
        """Return all RAG tools."""
        return [
            upload_pdf_document,
            upload_pdf_directory,
            query_documents,
            list_documents,
            delete_document,
            clear_knowledge_base,
        ]

