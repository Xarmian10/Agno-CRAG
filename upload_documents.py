"""
Admin script to upload PDF documents to the knowledge base.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_tools import get_vector_store, _upload_pdf_document_impl, _list_documents_impl


def upload_pdf(file_path: str, doc_id: str = None) -> None:
    """Upload a single PDF file."""
    print(f"Processing: {file_path}")
    result = _upload_pdf_document_impl(file_path=file_path, doc_id=doc_id)
    
    if result.get("success"):
        print(f"[成功] Uploaded successfully!")
        print(f"  - Doc ID: {result['doc_id']}")
        print(f"  - Total passages: {result['passage_count']}")
        if 'pages_with_content' in result:
            print(f"  - Pages with content: {result['pages_with_content']}")
        print(f"  - File: {result['file_path']}")
    else:
        print(f"[失败] Upload failed!")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(f"\n  Traceback:")
            print(result['traceback'])


def upload_directory(directory: str, pattern: str = "*.pdf", recursive: bool = False) -> None:
    """
    Upload all PDF files in a directory.
    
    Args:
        directory: Directory path containing PDF files
        pattern: File pattern to match (default: "*.pdf")
        recursive: If True, search subdirectories recursively
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[错误] Error: Directory not found: {directory}")
        return
    
    if not dir_path.is_dir():
        print(f"[错误] Error: Not a directory: {directory}")
        return
    
    # Find PDF files
    if recursive:
        pdf_files = list(dir_path.rglob(pattern))
    else:
        pdf_files = list(dir_path.glob(pattern))
    
    if not pdf_files:
        print(f"[警告] No PDF files found in {directory}")
        if recursive:
            print(f"   (searched recursively)")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s) in {directory}")
    if recursive:
        print(f"   (including subdirectories)")
    print("-" * 80)
    
    # Upload statistics
    success_count = 0
    fail_count = 0
    total_passages = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] ", end="")
        result = _upload_pdf_document_impl(file_path=str(pdf_file))
        
        if result.get("success"):
            success_count += 1
            total_passages += result.get("passage_count", 0)
            print(f"[成功] {pdf_file.name}")
            print(f"  Passages: {result.get('passage_count', 0)}")
        else:
            fail_count += 1
            print(f"[失败] {pdf_file.name}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Upload Summary:")
    print(f"  [成功] Successful: {success_count}")
    print(f"  [失败] Failed: {fail_count}")
    print(f"  Total passages: {total_passages}")
    print("=" * 80)


def clear_knowledge_base() -> None:
    """Clear all documents from the knowledge base."""
    from rag_tools import _clear_knowledge_base_impl
    
    print("[警告] WARNING: This will delete ALL documents and passages from the knowledge base!")
    print("   This action cannot be undone.\n")
    
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    result = _clear_knowledge_base_impl()
    
    if result.get("success"):
        print("\n[成功] Knowledge base cleared successfully!")
        print(f"  - Documents deleted: {result.get('documents_deleted', 0)}")
        print(f"  - Passages deleted: {result.get('passages_deleted', 0)}")
    else:
        print(f"\n[失败] Failed to clear knowledge base!")
        print(f"  Error: {result.get('error', 'Unknown error')}")


def list_all_documents() -> None:
    """List all documents in the knowledge base."""
    result = _list_documents_impl()
    if result.get("success"):
        docs = result.get("documents", [])
        if not docs:
            print("No documents in the knowledge base.")
            return
        
        print(f"\nDocuments in knowledge base ({result['count']}):")
        print("-" * 80)
        for doc in docs:
            print(f"  ID: {doc['doc_id']}")
            print(f"  Path: {doc['file_path']}")
            print(f"  Uploaded: {doc['uploaded_at']}")
            print()
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload PDF documents to the knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single PDF
  python upload_documents.py --file "document.pdf"
  
  # Upload all PDFs in a directory
  python upload_documents.py --dir "C:\\Users\\Documents\\PDFs"
  
  # Upload PDFs recursively (including subdirectories)
  python upload_documents.py --dir "C:\\Users\\Documents\\PDFs" --recursive
  
  # Upload with custom pattern
  python upload_documents.py --dir "C:\\Users\\Documents" --pattern "*.PDF"
  
  # List all documents
  python upload_documents.py --list
        """
    )
    parser.add_argument("--file", type=str, help="Path to a single PDF file to upload")
    parser.add_argument("--dir", type=str, help="Directory containing PDF files to upload")
    parser.add_argument("--doc-id", type=str, help="Custom document ID (only for --file)")
    parser.add_argument("--list", action="store_true", help="List all documents in the knowledge base")
    parser.add_argument("--pattern", type=str, default="*.pdf", help="File pattern for --dir (default: *.pdf)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search subdirectories recursively (only for --dir)")
    parser.add_argument("--clear", action="store_true", help="Clear all documents from the knowledge base (requires confirmation)")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_knowledge_base()
    elif args.list:
        list_all_documents()
    elif args.file:
        upload_pdf(args.file, args.doc_id)
    elif args.dir:
        upload_directory(args.dir, args.pattern, recursive=args.recursive)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

