"""
CRAG core components: semantic evaluator, web search, and action router.

This module consolidates all CRAG core functionality including:
- Semantic retrieval evaluator (T5-based)
- Web search augmenter
- Complete action router
- Enhanced knowledge refiner
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List, Literal, Optional, Tuple

from crag_layer import crag_evaluate_and_route, CragAction


def safe_print(message: str, flush: bool = True):
    """
    Safely print message handling Unicode encoding issues on Windows.
    Replaces emoji and special characters if encoding fails.
    """
    if not isinstance(message, str):
        message = str(message)
    
    try:
        print(message, flush=flush)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Replace problematic characters with ASCII equivalents
        try:
            safe_message = message.encode('ascii', errors='replace').decode('ascii', errors='replace')
            print(safe_message, flush=flush)
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

# Semantic Evaluator
try:
    import torch
    from transformers import T5ForSequenceClassification, T5Tokenizer
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False
    torch = None
    T5ForSequenceClassification = None
    T5Tokenizer = None

# Web Search
try:
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import quote_plus
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    requests = None
    BeautifulSoup = None


# ============================================================================
# Semantic Retrieval Evaluator
# ============================================================================

class SemanticRetrievalEvaluator:
    """
    T5-based semantic retrieval evaluator for CRAG.
    
    Evaluates the semantic relevance between a query and document using a
    fine-tuned T5 model. Returns continuous scores in the range [-1, 1].
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 4,
    ):
        """Initialize the semantic evaluator."""
        if not T5_AVAILABLE:
            raise RuntimeError(
                "T5 dependencies not available. Please install: "
                "pip install torch transformers sentencepiece"
            )
        
        if model_path is None:
            model_path = "finetuned_t5_evaluator"
            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"T5 model not found at {model_path}. "
                    "Please download the fine-tuned T5 evaluator model."
                )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.total_evaluations = 0
        self.total_time = 0.0
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            try:
                self.model = T5ForSequenceClassification.from_pretrained(
                    model_path, num_labels=1, use_safetensors=True,
                )
            except Exception:
                self.model = T5ForSequenceClassification.from_pretrained(
                    model_path, num_labels=1,
                )
            self.model.to(self.device)
            self.model.eval()
            
            if torch.cuda.is_available():
                safe_print(f"[完成] T5模型已加载到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                safe_print(f"[警告] T5模型加载到 CPU (GPU不可用，性能会较慢)")
        except Exception as e:
            raise RuntimeError(f"Failed to load T5 evaluator from {model_path}: {e}")
    
    def evaluate_relevance(self, query: str, document: str) -> float:
        """Evaluate semantic relevance between query and document."""
        if len(document.split()) < 4:
            return -1.0
        
        input_text = f"{query} [SEP] {document}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                )
            return float(outputs["logits"].cpu().item())
        except Exception:
            return -1.0
    
    def evaluate_batch(self, query: str, documents: List[str]) -> List[float]:
        """Evaluate relevance for multiple documents using TRUE batch processing."""
        if not documents:
            return []
        
        start_time = time.time()
        safe_print(f"\n  语义评估开始: {len(documents)} 个文档, 批处理大小: {self.batch_size}")
        
        # Filter valid documents
        filter_start = time.time()
        valid_docs = []
        valid_indices = []
        for i, doc in enumerate(documents):
            if len(doc.split()) >= 4:
                valid_docs.append(doc)
                valid_indices.append(i)
        
        if not valid_docs:
            safe_print(f"  [警告] 所有文档都被过滤（太短）")
            return [-1.0] * len(documents)
        
        safe_print(f"  有效文档: {len(valid_docs)}/{len(documents)}")
        filter_time = time.time() - filter_start
        
        # Prepare batch texts
        prep_start = time.time()
        batch_texts = [f"{query} [SEP] {doc}" for doc in valid_docs]
        prep_time = time.time() - prep_start
        
        # Batch processing
        all_scores = []
        num_batches = (len(batch_texts) + self.batch_size - 1) // self.batch_size
        safe_print(f"  批处理: {num_batches} 个批次")
        
        for i in range(0, len(batch_texts), self.batch_size):
            batch_start = time.time()
            batch = batch_texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            # Tokenize
            tokenize_start = time.time()
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            tokenize_time = time.time() - tokenize_start
            
            # Inference
            inference_start = time.time()
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = outputs["logits"].squeeze(-1).cpu().tolist()
                    if len(batch) == 1:
                        batch_scores = [batch_scores] if not isinstance(batch_scores, list) else batch_scores
                    all_scores.extend(batch_scores)
                inference_time = time.time() - inference_start
                safe_print(f"    批次 {batch_num}/{num_batches}: tokenize={tokenize_time:.3f}秒, "
                      f"inference={inference_time:.3f}秒, 总计={time.time() - batch_start:.3f}秒")
            except Exception as e:
                safe_print(f"    [警告] 批次 {batch_num} 处理失败: {e}")
                all_scores.extend([-1.0] * len(batch))
        
        # Reconstruct scores
        recon_start = time.time()
        full_scores = [-1.0] * len(documents)
        for idx, score in zip(valid_indices, all_scores):
            full_scores[idx] = float(score)
        recon_time = time.time() - recon_start
        
        elapsed = time.time() - start_time
        self.total_evaluations += len(documents)
        self.total_time += elapsed
        
        safe_print(f"  语义评估完成: {elapsed:.3f}秒")
        safe_print(f"     过滤: {filter_time:.3f}秒, 准备: {prep_time:.3f}秒, 重构: {recon_time:.3f}秒")
        safe_print(f"     平均每文档: {elapsed/len(documents):.3f}秒")
        
        return full_scores
    
    def get_scorer_function(self) -> Callable[[str, str], float]:
        """Get a scorer function compatible with crag_layer."""
        return self.evaluate_relevance
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        avg_time = self.total_time / self.total_evaluations if self.total_evaluations > 0 else 0.0
        return {
            "total_evaluations": self.total_evaluations,
            "total_time": self.total_time,
            "avg_time_per_doc": avg_time,
            "device": str(self.device),
            "batch_size": self.batch_size,
        }


# ============================================================================
# Web Search Augmenter
# ============================================================================

class QueryRewriter:
    """Query rewriter for better web search results."""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client
    
    def rewrite(self, query: str) -> str:
        """Rewrite query for better web search."""
        if self.use_llm and self.llm_client:
            return self._rewrite_with_llm(query)
        else:
            return self._rewrite_heuristic(query)
    
    def _rewrite_heuristic(self, query: str) -> str:
        """Heuristic-based query rewriting."""
        question_words = ["what", "who", "when", "where", "why", "how", "which"]
        words = query.lower().split()
        
        if len(words) > 5:
            words = [w for w in words if w not in question_words]
        
        rewritten = " ".join(words)
        if any(word in query.lower() for word in ["is", "was", "are", "definition", "meaning"]):
            rewritten = f"{rewritten} Wikipedia"
        
        return rewritten.strip()
    
    def _rewrite_with_llm(self, query: str) -> str:
        """LLM-based query rewriting."""
        if not self.llm_client:
            return self._rewrite_heuristic(query)
        return self._rewrite_heuristic(query)  # Fallback for now


class WebSearchAugmenter:
    """Web search augmenter for external knowledge retrieval."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine: str = "google",
        use_wikipedia: bool = True,
        max_results: int = 5,
    ):
        if not WEB_SEARCH_AVAILABLE:
            raise RuntimeError(
                "Web search dependencies not available. Please install: "
                "pip install beautifulsoup4 requests"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine = search_engine
        self.use_wikipedia = use_wikipedia
        self.max_results = max_results
        self.query_rewriter = QueryRewriter()
        self.api_calls = 0
        self.total_cost = 0.0
    
    def search_external_knowledge(self, query: str) -> List[Dict[str, str]]:
        """Execute web search and return relevant documents."""
        search_query = self.query_rewriter.rewrite(query)
        
        if self.search_engine == "google":
            results = self._search_google(search_query)
        elif self.search_engine == "bing":
            results = self._search_bing(search_query)
        elif self.search_engine == "duckduckgo":
            results = self._search_duckduckgo(search_query)
        else:
            raise ValueError(f"Unsupported search engine: {self.search_engine}")
        
        documents = self._extract_and_filter(results)
        if self.use_wikipedia:
            documents = self._prioritize_authoritative(documents)
        
        return documents[:self.max_results]
    
    def _search_google(self, query: str) -> List[Dict]:
        """Search using Google Custom Search API."""
        if not self.api_key:
            return self._search_duckduckgo(query)
        
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not search_engine_id:
            return self._search_duckduckgo(query)
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": search_engine_id,
            "q": query,
            "num": min(self.max_results * 2, 10),
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            self.api_calls += 1
            self.total_cost += 0.005
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })
            return results
        except Exception:
            return self._search_duckduckgo(query)
    
    def _search_bing(self, query: str) -> List[Dict]:
        """Search using Bing Search API."""
        if not self.api_key:
            return self._search_duckduckgo(query)
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": min(self.max_results * 2, 10)}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            self.api_calls += 1
            self.total_cost += 0.004
            
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                })
            return results
        except Exception:
            return self._search_duckduckgo(query)
    
    def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for result in soup.find_all("div", class_="result")[:self.max_results * 2]:
                title_elem = result.find("a", class_="result__a")
                snippet_elem = result.find("a", class_="result__snippet")
                
                if title_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "url": title_elem.get("href", ""),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                    })
            
            return results
        except Exception:
            return []
    
    def _extract_and_filter(self, results: List[Dict]) -> List[Dict[str, str]]:
        """Extract content from search results and filter by quality."""
        documents = []
        
        for result in results:
            content = result.get("snippet", "")
            url = result.get("url", "")
            
            if "wikipedia.org" in url.lower() or "edu" in url.lower():
                try:
                    full_content = self._fetch_page_content(url)
                    if full_content:
                        content = full_content
                except Exception:
                    pass
            
            if len(content) > 50:
                documents.append({
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", ""),
                    "content": content[:2000],
                })
        
        return documents
    
    def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a web page."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception:
            return None
    
    def _prioritize_authoritative(self, documents: List[Dict]) -> List[Dict]:
        """Prioritize authoritative sources (Wikipedia, .edu, .gov, etc.)."""
        authoritative = []
        others = []
        
        authoritative_domains = [
            "wikipedia.org", ".edu", ".gov", ".org",
            "scholar.google", "arxiv.org",
        ]
        
        for doc in documents:
            url = doc.get("url", "").lower()
            is_authoritative = any(domain in url for domain in authoritative_domains)
            
            if is_authoritative:
                authoritative.append(doc)
            else:
                others.append(doc)
        
        return authoritative + others
    
    def get_usage_stats(self) -> Dict[str, float]:
        """Get API usage statistics."""
        return {
            "api_calls": self.api_calls,
            "total_cost": self.total_cost,
        }


# ============================================================================
# Action Router and Knowledge Refiner
# ============================================================================

class EnhancedKnowledgeRefiner:
    """Enhanced knowledge refiner using semantic evaluation."""
    
    def __init__(self, evaluator: Optional[SemanticRetrievalEvaluator] = None):
        self.evaluator = evaluator
    
    def refine(
        self,
        query: str,
        documents: List[str],
        mode: str = "excerption",
        upper_threshold: float = 0.6,
        lower_threshold: float = 0.2,
        top_k: int = 5,
    ) -> Dict[str, object]:
        """Refine knowledge using semantic evaluation."""
        scorer_func = None
        score_range = (0.0, 1.0)
        
        if self.evaluator:
            scorer_func = self.evaluator.get_scorer_function()
            score_range = (-1.0, 1.0)
        
        result = crag_evaluate_and_route(
            query=query,
            candidate_passages=documents,
            top_k=top_k,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            decompose_mode=mode,
            scorer_func=scorer_func,
            score_range=score_range,
        )
        
        return result


class CompleteActionRouter:
    """Complete action router implementing the three-action routing mechanism."""
    
    def __init__(
        self,
        evaluator: Optional[SemanticRetrievalEvaluator] = None,
        web_searcher: Optional[WebSearchAugmenter] = None,
        upper_threshold: float = 0.6,
        lower_threshold: float = 0.2,
    ):
        self.evaluator = evaluator
        self.web_searcher = web_searcher
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.knowledge_refiner = EnhancedKnowledgeRefiner(evaluator)
        self.action_logs: List[Dict] = []
    
    def route_action(
        self,
        query: str,
        retrieved_docs: List[str],
        decompose_mode: str = "excerption",
    ) -> Dict[str, object]:
        """Complete three-action routing decision and execution."""
        import time
        route_start = time.time()
        
        safe_print(f"\n  动作路由开始: {len(retrieved_docs)} 个文档")
        
        # Evaluate retrieval quality
        eval_start = time.time()
        confidence_scores = self._evaluate_retrieval_quality(query, retrieved_docs)
        eval_time = time.time() - eval_start
        safe_print(f"  质量评估: {eval_time:.3f}秒")
        
        # Calculate global score
        score_start = time.time()
        global_score = self._calculate_global_score(confidence_scores)
        score_time = time.time() - score_start
        safe_print(f"  全局分数: {global_score:.4f} (计算耗时: {score_time:.3f}秒)")
        
        # Decide action
        action = self._decide_action(global_score)
        safe_print(f"  [完成] 决策动作: {action} (阈值: upper={self.upper_threshold}, lower={self.lower_threshold})")
        
        knowledge = None
        external_knowledge = None
        
        if action == "correct":
            result = self.knowledge_refiner.refine(
                query=query,
                documents=retrieved_docs,
                mode=decompose_mode,
                upper_threshold=self.upper_threshold,
                lower_threshold=self.lower_threshold,
            )
            knowledge = result.get("selected_strips", [])
        
        elif action == "incorrect":
            if self.web_searcher:
                external_docs = self.web_searcher.search_external_knowledge(query)
                knowledge = [doc.get("content", doc.get("snippet", "")) for doc in external_docs]
            else:
                result = self.knowledge_refiner.refine(
                    query=query,
                    documents=retrieved_docs,
                    mode=decompose_mode,
                )
                knowledge = result.get("selected_strips", [])
        
        elif action == "ambiguous":
            internal_result = self.knowledge_refiner.refine(
                query=query,
                documents=retrieved_docs,
                mode=decompose_mode,
            )
            internal_knowledge = internal_result.get("selected_strips", [])
            
            if self.web_searcher:
                external_docs = self.web_searcher.search_external_knowledge(query)
                external_knowledge = [doc.get("content", doc.get("snippet", "")) for doc in external_docs]
            else:
                external_knowledge = []
            
            knowledge = self._combine_knowledge(internal_knowledge, external_knowledge)
        
        route_time = time.time() - route_start
        safe_print(f"  动作路由总耗时: {route_time:.3f}秒")
        
        log_entry = {
            "query": query,
            "action": action,
            "global_score": global_score,
            "num_internal_docs": len(retrieved_docs),
            "num_knowledge_strips": len(knowledge) if knowledge else 0,
            "has_external": external_knowledge is not None and len(external_knowledge) > 0,
        }
        self.action_logs.append(log_entry)
        
        return {
            "action": action,
            "knowledge": knowledge or [],
            "global_score": global_score,
            "internal_knowledge": retrieved_docs,
            "external_knowledge": external_knowledge,
            "debug": {
                "confidence_scores": confidence_scores,
                "num_retrieved": len(retrieved_docs),
                "num_knowledge_strips": len(knowledge) if knowledge else 0,
            },
        }
    
    def _evaluate_retrieval_quality(
        self,
        query: str,
        retrieved_docs: List[str],
    ) -> List[float]:
        """Evaluate retrieval quality for each document."""
        if not retrieved_docs:
            return []
        
        if self.evaluator:
            scores = self.evaluator.evaluate_batch(query, retrieved_docs)
        else:
            from crag_layer import _simple_overlap_score
            scores = [_simple_overlap_score(query, doc) for doc in retrieved_docs]
        
        return scores
    
    def _calculate_global_score(self, confidence_scores: List[float]) -> float:
        """Calculate global retrieval quality score."""
        if not confidence_scores:
            return 0.0
        
        sorted_scores = sorted(confidence_scores, reverse=True)
        top_k = min(5, len(sorted_scores))
        return sum(sorted_scores[:top_k]) / top_k
    
    def _decide_action(self, global_score: float) -> CragAction:
        """Decide action based on global score and thresholds."""
        if self.evaluator:
            normalized_score = (global_score + 1.0) / 2.0
        else:
            normalized_score = global_score
        
        if normalized_score >= self.upper_threshold:
            return "correct"
        elif normalized_score >= self.lower_threshold:
            return "ambiguous"
        else:
            return "incorrect"
    
    def _combine_knowledge(
        self,
        internal_knowledge: List[str],
        external_knowledge: List[str],
    ) -> List[str]:
        """Combine internal and external knowledge for ambiguous action."""
        combined = []
        combined.extend(internal_knowledge)
        combined.extend(external_knowledge)
        return combined[:10]
    
    def get_action_stats(self) -> Dict[str, int]:
        """Get statistics about action routing."""
        stats = {"correct": 0, "incorrect": 0, "ambiguous": 0}
        for log in self.action_logs:
            action = log.get("action", "ambiguous")
            stats[action] = stats.get(action, 0) + 1
        return stats

