from __future__ import annotations

"""
Lightweight CRAG-style utilities integrated as an Agno tool.

This module implements a simplified, model-agnostic version of the
Corrective Retrieval-Augmented Generation (CRAG) logic, inspired by:
`Corrective Retrieval Augmented Generation` (Yan et al., 2024).

Core features:
- A retrieval evaluator that scores candidate passages for a query.
- Decompose-then-recompose: split passages into strips and select top-k.
- Action routing with three actions: Correct / Incorrect / Ambiguous.

This implementation is intentionally lightweight:
- It does NOT load any external transformer models.
- It uses simple lexical similarity heuristics instead of a T5 evaluator.
- The logic is used internally by rag_tools.py for CRAG evaluation.
"""

from dataclasses import dataclass
import sys
from typing import Callable, Dict, List, Literal, Optional, Tuple


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

try:
    import torch
    from transformers import T5ForSequenceClassification, T5Tokenizer
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False
    torch = None
    T5ForSequenceClassification = None
    T5Tokenizer = None


CragAction = Literal["correct", "incorrect", "ambiguous"]


@dataclass
class StripScore:
    score: float
    text: str
    passage_index: int
    strip_index: int


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _simple_overlap_score(query: str, text: str) -> float:
    """
    Compute a simple lexical overlap score between query and text.

    This is a lightweight surrogate for the T5-based evaluator in CRAG.
    It approximates relevance in a cheap way and returns a score in [0, 1].
    """

    q_tokens = set(_normalize_text(query).split())
    t_tokens = set(_normalize_text(text).split())

    if not q_tokens or not t_tokens:
        return 0.0

    intersection = q_tokens.intersection(t_tokens)
    union = q_tokens.union(t_tokens)
    return len(intersection) / len(union)


def _split_into_strips(
    passage: str, 
    mode: str = "excerption",
    max_sentences_per_strip: int = 3,
    window_length: int = 50,
) -> List[str]:
    """
    Decompose a passage into smaller strips, following CRAG paper implementation.
    
    Args:
        passage: The text passage to decompose.
        mode: One of 'fixed_num', 'excerption', or 'selection'.
            - 'fixed_num': Segment by fixed number of words (window_length).
            - 'excerption': Segment by sentences, grouping up to max_sentences_per_strip.
            - 'selection': Return the passage as-is (no decomposition).
        max_sentences_per_strip: For 'excerption' mode, max sentences per strip.
        window_length: For 'fixed_num' mode, number of words per strip.
    
    Returns:
        List of text strips.
    """
    if not passage:
        return []
    
    if mode == "selection":
        return [passage]
    
    if mode == "fixed_num":
        # Segment by fixed number of words (from paper: window_length=50)
        final_strips = []
        words = passage.split()
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) == window_length:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            if len(buf) < 10 and final_strips:
                # Merge short remainder with last strip
                final_strips[-1] += " " + " ".join(buf)
            else:
                final_strips.append(" ".join(buf))
        return final_strips if final_strips else [passage]
    
    if mode == "excerption":
        # Segment by sentences, then group (from paper: num_concatenate_strips=3)
        num_concatenate_strips = max_sentences_per_strip
        question_strips = passage.split("?")
        origin_strips = []
        for qs in question_strips:
            origin_strips.extend(qs.split(". "))
        
        strips = []
        for s in origin_strips:
            if s in strips:
                continue
            if not strips:
                strips.append(s)
            else:
                if len(s.split()) > 5:
                    strips.append(s)
                else:
                    # Merge short sentences with previous strip
                    strips[-1] += " " + s
        
        final_strips = []
        buf = []
        for strip in strips:
            buf.append(strip)
            if len(buf) == num_concatenate_strips:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            final_strips.append(" ".join(buf))
        return final_strips if final_strips else [passage]
    
    # Fallback: simple sentence split
    separators = [".", "!", "?", "。", "！", "？"]
    tmp = passage
    for sep in separators:
        tmp = tmp.replace(sep, ".")
    sentences = [s.strip() for s in tmp.split(".") if s.strip()]
    if not sentences:
        return [passage.strip()]
    strips = []
    current = []
    for sent in sentences:
        current.append(sent)
        if len(current) >= max_sentences_per_strip:
            strips.append(" ".join(current))
            current = []
    if current:
        strips.append(" ".join(current))
    return strips if strips else [passage.strip()]


def _score_strips(
    query: str, 
    passages: List[str],
    decompose_mode: str = "excerption",
    scorer_func=None,
) -> List[StripScore]:
    """
    Score strips from passages using the specified decompose mode.
    
    Args:
        query: User query.
        passages: List of candidate passages.
        decompose_mode: 'fixed_num', 'excerption', or 'selection'.
        scorer_func: Optional custom scoring function (query, strip) -> float.
            If None, uses _simple_overlap_score. For T5 evaluator, pass a function
            that calls the T5 model.
    
    Returns:
        List of StripScore objects.
    """
    if scorer_func is None:
        scorer_func = _simple_overlap_score
    
    scored: List[StripScore] = []
    for p_idx, passage in enumerate(passages):
        strips = _split_into_strips(passage, mode=decompose_mode)
        for s_idx, strip in enumerate(strips):
            score = scorer_func(query, strip)
            scored.append(
                StripScore(
                    score=score,
                    text=strip,
                    passage_index=p_idx,
                    strip_index=s_idx,
                )
            )
    return scored


def _decide_action(
    global_score: float, 
    upper_threshold: float, 
    lower_threshold: float,
    score_range: Tuple[float, float] = (0.0, 1.0),
) -> CragAction:
    """
    Decide CRAG action based on a global score, following paper implementation.
    
    In the original CRAG paper:
    - Evaluator outputs logits (can be negative, e.g., [-1, 1] after normalization)
    - threshold1 (upper) and threshold2 (lower) are dataset-specific
    - score >= threshold1 -> 'correct' (flag=2)
    - score >= threshold2 -> 'ambiguous' (flag=1)  
    - score < threshold2 -> 'incorrect' (flag=0)
    
    Args:
        global_score: Overall relevance score.
        upper_threshold: Threshold for 'correct' action.
        lower_threshold: Threshold for 'ambiguous' action (below this is 'incorrect').
        score_range: (min, max) of the score range. Default (0.0, 1.0) for normalized scores.
            For T5 logits, use (-1.0, 1.0) or actual logit range.
    
    Returns:
        One of 'correct', 'incorrect', 'ambiguous'.
    """
    # Normalize thresholds if score_range is different from [0, 1]
    min_score, max_score = score_range
    if min_score != 0.0 or max_score != 1.0:
        # Map score to [0, 1] for comparison
        normalized_score = (global_score - min_score) / (max_score - min_score) if max_score != min_score else 0.0
        normalized_upper = (upper_threshold - min_score) / (max_score - min_score) if max_score != min_score else 0.0
        normalized_lower = (lower_threshold - min_score) / (max_score - min_score) if max_score != min_score else 0.0
    else:
        normalized_score = global_score
        normalized_upper = upper_threshold
        normalized_lower = lower_threshold
    
    # Paper logic: score >= upper -> correct, score >= lower -> ambiguous, else incorrect
    if normalized_score >= normalized_upper:
        return "correct"
    if normalized_score >= normalized_lower:
        return "ambiguous"
    return "incorrect"


def crag_evaluate_and_route(
    query: str,
    candidate_passages: List[str],
    top_k: int = 5,
    upper_threshold: float = 0.6,
    lower_threshold: float = 0.2,
    decompose_mode: str = "excerption",
    scorer_func=None,
    score_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, object]:
    """
    Evaluate retrieval quality and perform decompose-then-recompose on passages.
    Follows the CRAG paper implementation.

    Args:
        query: User question or instruction.
        candidate_passages: A list of raw passages retrieved from an internal
            corpus or search engine. These can be long paragraphs.
        top_k: Number of strips to keep after scoring (default: 5).
            For 'selection' mode, top_n=3; for other modes, top_n=6 (per paper).
        upper_threshold: Threshold for 'correct' action (paper: threshold1).
        lower_threshold: Threshold for 'ambiguous' action (paper: threshold2).
        decompose_mode: 'fixed_num', 'excerption', or 'selection' (default: 'excerption').
        scorer_func: Optional custom scoring function (query, strip) -> float.
            If None, uses _simple_overlap_score. For T5 evaluator, pass a function
            that calls T5ForSequenceClassification.
        score_range: (min, max) of score range. Default (0.0, 1.0) for normalized scores.
            For T5 logits, use (-1.0, 1.0) or actual logit range.

    Returns:
        A dict with:
        - action: one of {'correct', 'incorrect', 'ambiguous'}.
        - global_score: overall relevance score.
        - selected_strips: top-k strips (short text segments) for generation.
        - debug: optional debug info (scores, indices) that the agent can inspect.

    Notes:
        - This function follows the CRAG paper's decompose-then-recompose algorithm.
        - For T5 evaluator integration, pass a scorer_func that wraps T5 model calls.
        - External web search for 'incorrect' or 'ambiguous' actions can be added
          by the caller, using the returned action as a signal.
    """
    import time
    crag_start = time.time()
    verbose = os.getenv("VERBOSE_CRAG", "false").lower() == "true"
    
    if verbose:
        print(f"\n  🔄 CRAG评估开始: {len(candidate_passages)} 个段落, mode={decompose_mode}")
    
    if not candidate_passages:
        return {
            "action": "ambiguous",
            "global_score": 0.0,
            "selected_strips": [],
            "debug": {"reason": "no_passages"},
        }

    # Adjust top_k based on decompose_mode (per paper: selection=3, others=6)
    effective_top_k = 3 if decompose_mode == "selection" else min(top_k, 6)
    
    if verbose:
        safe_print(f"  有效top_k: {effective_top_k}, 评分函数: {'语义评估器' if scorer_func else '词法评分'}")
    
    # Score strips
    score_start = time.time()
    scored_strips = _score_strips(
        query, 
        candidate_passages, 
        decompose_mode=decompose_mode,
        scorer_func=scorer_func,
    )
    score_time = time.time() - score_start
    
    if verbose:
        safe_print(f"  片段评分: {score_time:.3f}秒, 生成 {len(scored_strips)} 个片段")
    
    if not scored_strips:
        return {
            "action": "ambiguous",
            "global_score": 0.0,
            "selected_strips": [],
            "debug": {"reason": "no_strips"},
        }

    # Compute global score as the average of top-k strip scores (paper approach)
    sort_start = time.time()
    sorted_by_score = sorted(scored_strips, key=lambda s: s.score, reverse=True)
    top_for_global = sorted_by_score[: max(1, min(effective_top_k, len(sorted_by_score)))]
    global_score = sum(s.score for s in top_for_global) / len(top_for_global)
    sort_time = time.time() - sort_start

    if verbose:
        safe_print(f"  排序和全局分数计算: {sort_time:.3f}秒")
        safe_print(f"  全局分数: {global_score:.4f} (top-{len(top_for_global)} 平均)")

    # Decide action using paper's logic
    action = _decide_action(
        global_score, 
        upper_threshold, 
        lower_threshold,
        score_range=score_range,
    )
    
    if verbose:
        safe_print(f"  [完成] 决策动作: {action}")

    # Decompose-then-recompose: keep only the top-k strips as key knowledge
    selected = sorted_by_score[: max(1, min(effective_top_k, len(sorted_by_score)))]
    selected_texts = [s.text for s in selected]
    
    if verbose:
        crag_time = time.time() - crag_start
        safe_print(f"  CRAG评估完成: {crag_time:.3f}秒, 选择 {len(selected_texts)} 个片段")

    debug_info: Dict[str, object] = {
        "num_passages": len(candidate_passages),
        "num_strips": len(scored_strips),
        "decompose_mode": decompose_mode,
        "top_scores": [round(s.score, 4) for s in selected],
        "top_indices": [
            {"passage_index": s.passage_index, "strip_index": s.strip_index}
            for s in selected
        ],
    }

    return {
        "action": action,
        "global_score": global_score,
        "selected_strips": selected_texts,
        "debug": debug_info,
    }


def create_t5_scorer(
    model_path: str,
    device: Optional[str] = None,
    max_length: int = 512,
) -> Optional[Callable[[str, str], float]]:
    """
    Create a T5-based scoring function for CRAG, following the paper implementation.
    
    This function wraps T5ForSequenceClassification to score (query, strip) pairs,
    matching the evaluator used in the CRAG paper.
    
    Args:
        model_path: Path to the fine-tuned T5 evaluator model (from paper's download link).
        device: Device to run the model on ('cuda', 'cpu', or None for auto-detect).
        max_length: Maximum sequence length for tokenization (default: 512).
    
    Returns:
        A scoring function (query: str, strip: str) -> float, or None if T5 is not available.
    
    Example:
        >>> scorer = create_t5_scorer("path/to/t5-evaluator")
        >>> result = crag_evaluate_and_route(
        ...     query="...",
        ...     candidate_passages=[...],
        ...     scorer_func=scorer,
        ...     score_range=(-1.0, 1.0),  # T5 logits can be negative
        ... )
    """
    if not T5_AVAILABLE:
        return None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        # Handle sharded safetensors models (model-001.safetensors, model-002.safetensors, etc.)
        # Transformers should auto-detect, but we can also explicitly use safetensors
        try:
            model = T5ForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=1,
                use_safetensors=True,
            )
        except Exception:
            # Fallback: try loading without explicit safetensors flag
            model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=1)
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load T5 evaluator from {model_path}: {e}")
    
    def t5_scorer(query: str, strip: str) -> float:
        """Score a (query, strip) pair using T5 evaluator."""
        if len(strip.split()) < 4:
            return -1.0  # Filter very short strips (per paper utils.py)
        
        input_content = f"{query} [SEP] {strip}"
        inputs = tokenizer(
            input_content,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        
        try:
            with torch.no_grad():
                outputs = model(
                    inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                )
            score = float(outputs["logits"].cpu().item())
            return score
        except Exception:
            return -1.0
    
    return t5_scorer

# Note: crag_evaluate_and_route is used directly by rag_tools.py
# No need to expose it as a tool since query_documents tool handles it

