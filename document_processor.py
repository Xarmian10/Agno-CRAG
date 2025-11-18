"""
Document processing: PDF reading and passage extraction.

This module consolidates all PDF and document processing functionality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agno.knowledge.chunking.document import DocumentChunking
from agno.knowledge.reader.pdf_reader import PDFReader


# Instantiate a reusable PDF reader that chunks by page
_pdf_reader = PDFReader(
    split_on_pages=True,
    chunking_strategy=DocumentChunking(chunk_size=4000),
)


@dataclass
class Passage:
    """Represents a text passage from a document."""
    doc_id: str
    page: Optional[int]
    text: str


def _format_pages(pages: List[Dict[str, Optional[str]]]) -> str:
    """Format pages into a single text string."""
    formatted_segments: List[str] = []
    for page in pages:
        page_number = page.get("page")
        prefix = f"[Page {page_number}]\n" if page_number is not None else ""
        formatted_segments.append(f"{prefix}{page.get('content', '')}".strip())
    return "\n\n".join(segment for segment in formatted_segments if segment)


def read_local_pdf(
    path: str,
    max_pages: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> Dict[str, object]:
    """
    Read a local PDF file and return its text content.

    Args:
        path: Absolute or relative path to a PDF file.
        max_pages: Optional limit on the number of pages returned (from the start).
        max_chars: Optional limit on the combined text length returned.

    Returns:
        Dict containing:
            - file: Absolute path to the PDF.
            - page_count: Number of pages returned.
            - pages: List of {page: int | None, content: str}.
            - combined_text: Single string concatenating page contents (respecting max_chars).
    """
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    documents = _pdf_reader.read(str(pdf_path))
    if max_pages is not None:
        documents = documents[: max(0, max_pages)]

    pages: List[Dict[str, Optional[str]]] = []
    for doc in documents:
        page_number = doc.meta_data.get("page") if doc.meta_data else None
        pages.append(
            {
                "page": int(page_number) if page_number is not None else None,
                "content": doc.content.strip() if doc.content else "",
            }
        )

    combined_text = _format_pages(pages)
    if max_chars is not None and len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]

    return {
        "file": str(pdf_path),
        "page_count": len(pages),
        "pages": pages,
        "combined_text": combined_text,
    }


class DocumentProcessor:
    """Processes PDF files and extracts structured passages."""

    def process_pdf(self, path: str) -> Tuple[str, List[Passage]]:
        """
        Process a PDF file and extract passages.
        
        Args:
            path: Path to the PDF file
            
        Returns:
            Tuple of (doc_id, list of Passage objects)
        """
        result = read_local_pdf(path)
        doc_id = Path(path).stem
        passages: List[Passage] = []
        for page in result.get("pages", []):
            content = (page.get("content") or "").strip()
            if not content:
                continue
            passages.append(
                Passage(
                    doc_id=doc_id,
                    page=page.get("page"),
                    text=content,
                )
            )
        return doc_id, passages

# Note: read_local_pdf is used internally by _upload_pdf_document_impl
# No need to expose it as a separate tool

