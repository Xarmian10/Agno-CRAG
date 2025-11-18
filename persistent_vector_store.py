"""
Persistent vector store using SQLite for document storage and retrieval.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from document_processor import Passage


def _extract_keywords(query: str) -> list:
    """
    Extract important keywords from a query, filtering out common stop words.
    Supports both English and Chinese queries.
    Improved to handle compound technical terms.
    """
    import re
    
    # Check if query contains Chinese characters
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    
    if has_chinese:
        # Chinese query: extract Chinese characters and numbers
        # First, try to extract compound technical terms (3+ characters)
        # Common patterns: "牵引索安全系数", "平衡索安全系数", etc.
        compound_terms = re.findall(r'[\u4e00-\u9fff]{3,}', query)
        
        # Extract Chinese words (2+ characters)
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        
        # Extract numbers (including standard numbers like GB19402-2012)
        numbers = re.findall(r'\d+[.\-\d]*', query)
        
        # Also extract single important Chinese characters
        single_chars = re.findall(r'[\u4e00-\u9fff]', query)
        
        # Combine: prefer compound terms, then multi-character words, then numbers, then single chars
        keywords = []
        
        # Add compound terms first (longer terms are more specific)
        for term in sorted(compound_terms, key=len, reverse=True):
            if term not in keywords:
                keywords.append(term)
        
        # Add 2+ character words that aren't already in compound terms
        for word in chinese_words:
            # Check if this word is part of any compound term
            is_part_of_compound = any(word in comp for comp in compound_terms)
            if not is_part_of_compound and word not in keywords:
                keywords.append(word)
        
        # Add numbers
        keywords.extend(numbers)
        
        # Add single chars only if they appear multiple times or are technical terms
        char_freq = {}
        for char in single_chars:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Add frequently appearing single chars or important technical terms
        important_chars = ['索', '车', '力', '数', '系', '规', '范', '标', '准', '引', '衡', '全']
        for char, freq in char_freq.items():
            if freq >= 2 or char in important_chars:
                if char not in keywords:
                    keywords.append(char)
        
        return keywords if keywords else single_chars[:5]  # Limit single chars
    
    else:
        # English query: use original logic
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'some', 'any', 'no', 'not', 'if', 'then', 'else', 'so', 'than', 'too',
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            keywords = [w for w in words if len(w) > 2]
        
        return keywords if keywords else words


def _lexical_overlap(query: str, text: str) -> float:
    """
    Calculate lexical overlap between query and text.
    Supports both English and Chinese queries with optimized matching strategies.
    """
    import re
    
    # Ensure Unicode strings and normalize
    if not isinstance(query, str):
        query = str(query, 'utf-8', errors='ignore')
    if not isinstance(text, str):
        text = str(text, 'utf-8', errors='ignore')
    
    # For Chinese, don't use lower() as it doesn't affect Chinese characters
    # and may cause issues with mixed content
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    
    if has_chinese:
        # For Chinese queries, work with original case (Chinese has no case)
        query_normalized = query.strip()
        text_normalized = text
    else:
        # For English queries, use lowercase
        query_normalized = query.lower().strip()
        text_normalized = text.lower()
    
    # Strategy 1: Check if full query appears as substring (highest priority)
    if query_normalized in text_normalized:
        return 0.8  # High score for exact phrase match
    
    if has_chinese:
        # Chinese query: character-level matching
        # Strategy 2: Extract Chinese keywords (use original case for Chinese)
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', query_normalized)
        numbers = re.findall(r'\d+', query_normalized)
        query_keywords = chinese_words + numbers
        
        if not query_keywords:
            # Fallback: use individual Chinese characters
            query_chars = list(set(re.findall(r'[\u4e00-\u9fff]', query_normalized)))
            query_keywords = query_chars
        
        # Count matches in text (use normalized text for Chinese)
        matched_keywords = 0
        total_matches = 0
        
        for keyword in query_keywords:
            count = text_normalized.count(keyword)
            if count > 0:
                matched_keywords += 1
                total_matches += count
        
        if matched_keywords == 0:
            return 0.0
        
        # Calculate score based on keyword match ratio and frequency
        keyword_ratio = matched_keywords / len(query_keywords)
        frequency_bonus = min(total_matches * 0.05, 0.3)
        
        score = keyword_ratio * 0.5 + frequency_bonus
        
        # Boost for longer matching phrases
        if len(chinese_words) > 0:
            for word in chinese_words:
                if len(word) >= 3 and word in text_normalized:
                    score = min(score + 0.2, 1.0)
        
        return min(score, 1.0)
    
    else:
        # English query: word-level matching (original logic)
        q_tokens = query_normalized.split()
        
        # For single word queries
        if len(q_tokens) == 1:
            word = q_tokens[0]
            count = text_normalized.count(word)
            if count > 0:
                base_score = 0.1
                frequency_bonus = min(count * 0.05, 0.4)
                return base_score + frequency_bonus
            return 0.0
        
        # For multi-word queries
        t_tokens = set(text_normalized.split())
        q_tokens_set = set(q_tokens)
        
        if not q_tokens_set or not t_tokens:
            return 0.0
        
        intersection = q_tokens_set & t_tokens
        
        if not intersection:
            return 0.0
        
        # Jaccard similarity
        union = q_tokens_set | t_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Keyword-based matching
        keywords = _extract_keywords(query)
        if keywords:
            keyword_set = set(keywords)
            keyword_intersection = keyword_set & t_tokens
            if keyword_intersection:
                keyword_ratio = len(keyword_intersection) / len(keyword_set)
                jaccard = max(jaccard, keyword_ratio * 0.7)
        
        # Boost based on match ratio
        match_ratio = len(intersection) / len(q_tokens_set)
        if match_ratio >= 0.5:
            jaccard = min(jaccard * (1.0 + match_ratio), 1.0)
        
        # Bonus for all keywords found
        if keywords and len(set(keywords) & t_tokens) == len(set(keywords)):
            jaccard = min(jaccard * 1.3, 1.0)
        
        return jaccard


class PersistentVectorStore:
    """A persistent vector store using SQLite for document storage."""

    def __init__(self, db_path: str = "rag_database.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        # Ensure UTF-8 encoding for SQLite
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Set text factory to handle unicode properly
        conn.text_factory = str
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Passages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page INTEGER,
                text TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        
        # Create indexes for faster search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_id ON passages(doc_id)
        """)
        
        conn.commit()
        conn.close()

    def add_document(self, doc_id: str, file_path: str, passages: Sequence[Passage], metadata: Optional[Dict] = None) -> None:
        """Add a document and its passages to the store."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str  # Ensure UTF-8 handling
        cursor = conn.cursor()
        
        try:
            # Insert or update document
            metadata_json = json.dumps(metadata or {})
            cursor.execute("""
                INSERT OR REPLACE INTO documents (doc_id, file_path, metadata)
                VALUES (?, ?, ?)
            """, (doc_id, str(file_path), metadata_json))
            
            # Delete existing passages for this document
            cursor.execute("DELETE FROM passages WHERE doc_id = ?", (doc_id,))
            
            # Insert new passages
            for passage in passages:
                # Ensure all values are proper Unicode strings
                doc_id_str = str(passage.doc_id) if passage.doc_id else ""
                text_str = str(passage.text) if passage.text else ""
                
                # Verify encoding
                if isinstance(text_str, bytes):
                    text_str = text_str.decode('utf-8', errors='ignore')
                if isinstance(doc_id_str, bytes):
                    doc_id_str = doc_id_str.decode('utf-8', errors='ignore')
                
                cursor.execute("""
                    INSERT INTO passages (doc_id, page, text)
                    VALUES (?, ?, ?)
                """, (doc_id_str, passage.page, text_str))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to add document {doc_id}: {e}")
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doc_id, file_path, uploaded_at, metadata
            FROM documents
            WHERE doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "doc_id": row[0],
            "file_path": row[1],
            "uploaded_at": row[2],
            "metadata": json.loads(row[3]) if row[3] else {},
        }

    def list_documents(self) -> List[Dict]:
        """List all documents."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doc_id, file_path, uploaded_at, metadata
            FROM documents
            ORDER BY uploaded_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "doc_id": row[0],
                "file_path": row[1],
                "uploaded_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
            }
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its passages."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM passages WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to delete document {doc_id}: {e}")
        finally:
            conn.close()
        
        return deleted

    def search(
        self, 
        query: str, 
        similarity_threshold: float = 0.15, 
        top_k: int = 12,
        doc_id_filter: Optional[str] = None,
    ) -> List[Tuple[float, Passage]]:
        """
        Search for relevant passages.
        
        Args:
            query: Search query string
            similarity_threshold: Minimum similarity score
            top_k: Maximum number of results to return
            doc_id_filter: Optional document ID or prefix to filter results (e.g., "GB146" will match "GB146.1-2020")
        """
        import time
        search_start = time.time()
        
        # Database query
        db_start = time.time()
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str  # Ensure UTF-8 handling for Chinese text
        cursor = conn.cursor()
        
        # Get all passages, optionally filtered by doc_id
        # Ensure doc_id_filter is a proper string
        if doc_id_filter:
            doc_id_filter_str = str(doc_id_filter)
            # Use parameterized query to avoid encoding issues
            cursor.execute("""
                SELECT doc_id, page, text 
                FROM passages 
                WHERE doc_id LIKE ? || '%'
            """, (doc_id_filter_str,))
        else:
            cursor.execute("SELECT doc_id, page, text FROM passages")
        
        rows = cursor.fetchall()
        conn.close()
        db_time = time.time() - db_start
        
        # Convert to Passage objects and score
        convert_start = time.time()
        passages = []
        for row in rows:
            # Ensure all text is properly decoded as Unicode
            try:
                doc_id = row[0] if row[0] is not None else ""
                page = row[1] if row[1] is not None else None
                text = row[2] if row[2] is not None else ""
                
                # Ensure Unicode strings
                if not isinstance(doc_id, str):
                    doc_id = str(doc_id, 'utf-8', errors='ignore') if isinstance(doc_id, bytes) else str(doc_id)
                if not isinstance(text, str):
                    text = str(text, 'utf-8', errors='ignore') if isinstance(text, bytes) else str(text)
                
                # Clean PDF encoding artifacts (e.g., /G21, /G22 from PDF font encoding)
                if text:
                    import re
                    # Remove PDF font encoding artifacts like /G21, /G22, etc.
                    text = re.sub(r'/[Gg][0-9A-Fa-f]{2}\s*', '', text)
                    # Remove other common PDF encoding artifacts
                    text = re.sub(r'\\[0-9]{3}', '', text)  # Remove octal escapes
                    # Clean up multiple spaces but preserve newlines
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = text.strip()
                
                if text:  # Only add non-empty cleaned text
                    passages.append(Passage(doc_id=doc_id, page=page, text=text))
            except Exception as e:
                # Skip problematic rows
                continue
        convert_time = time.time() - convert_start
        
        # Score passages
        score_start = time.time()
        scored: List[Tuple[float, Passage]] = []
        top_scores_debug = []  # For debugging
        
        for passage in passages:
            score = _lexical_overlap(query, passage.text)
            
            # Boost score if doc_id matches the query pattern
            if not doc_id_filter:
                # Try to detect document ID pattern in query (e.g., "GB146", "GB10493")
                query_upper = query.upper()
                if passage.doc_id.upper().startswith(query_upper) or query_upper in passage.doc_id.upper():
                    score = min(score * 1.5, 1.0)  # Boost score by 50%
            
            # Always track top scores for debugging (even if below threshold)
            if len(top_scores_debug) < 10:
                top_scores_debug.append((score, passage.doc_id, passage.page))
                top_scores_debug.sort(key=lambda x: x[0], reverse=True)
            elif score > top_scores_debug[-1][0]:
                top_scores_debug[-1] = (score, passage.doc_id, passage.page)
                top_scores_debug.sort(key=lambda x: x[0], reverse=True)
            
            if score >= similarity_threshold:
                scored.append((score, passage))
        
        score_time = time.time() - score_start
        
        # Debug output if no results found
        if len(scored) == 0 and os.getenv("VERBOSE_SEARCH", "false").lower() == "true":
            print(f"    [调试] 未找到结果，显示前10个最高分数:")
            for i, (score, doc_id, page) in enumerate(top_scores_debug[:5], 1):
                print(f"      {i}. 分数: {score:.4f}, 文档: {doc_id}, 页码: {page}")
            print(f"    [调试] 当前阈值: {similarity_threshold}, 段落总数: {len(passages)}")
        
        # Sort and limit
        sort_start = time.time()
        scored.sort(key=lambda item: item[0], reverse=True)
        result = scored[:top_k]
        sort_time = time.time() - sort_start
        
        total_time = time.time() - search_start
        
        # Detailed logging (only if verbose)
        if os.getenv("VERBOSE_SEARCH", "false").lower() == "true":
            print(f"    [搜索详情] 数据库查询: {db_time:.3f}秒, 转换: {convert_time:.3f}秒, "
                  f"评分: {score_time:.3f}秒 ({len(passages)}个段落), 排序: {sort_time:.3f}秒")
            print(f"    [搜索详情] 总耗时: {total_time:.3f}秒, 找到 {len(result)}/{len(scored)} 个结果")
        
        return result

    def get_passages_by_doc(self, doc_id: str) -> List[Passage]:
        """Get all passages for a specific document."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doc_id, page, text
            FROM passages
            WHERE doc_id = ?
            ORDER BY page
        """, (doc_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        passages = []
        for row in rows:
            try:
                doc_id = str(row[0]) if row[0] is not None else ""
                page = row[1] if row[1] is not None else None
                text = str(row[2]) if row[2] is not None else ""
                
                # Ensure Unicode strings
                if not isinstance(doc_id, str):
                    doc_id = str(doc_id, 'utf-8', errors='ignore') if isinstance(doc_id, bytes) else str(doc_id)
                if not isinstance(text, str):
                    text = str(text, 'utf-8', errors='ignore') if isinstance(text, bytes) else str(text)
                
                passages.append(Passage(doc_id=doc_id, page=page, text=text))
            except Exception:
                continue
        
        return passages

    def clear_all(self) -> Dict[str, int]:
        """
        Clear all documents and passages from the database.
        
        Returns:
            Dictionary with counts of deleted documents and passages.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.text_factory = str
        cursor = conn.cursor()
        
        try:
            # Count before deletion
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM passages")
            passage_count = cursor.fetchone()[0]
            
            # Delete all data
            cursor.execute("DELETE FROM passages")
            cursor.execute("DELETE FROM documents")
            
            conn.commit()
            
            return {
                "documents_deleted": doc_count,
                "passages_deleted": passage_count,
            }
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to clear database: {e}")
        finally:
            conn.close()

