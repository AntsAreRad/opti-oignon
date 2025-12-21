#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG CLI - Command Line Interface for Opti-Oignon RAG
====================================================

Usage:
    python -m opti_oignon.rag index ~/Documents/code
    python -m opti_oignon.rag search "Shannon index"
    python -m opti_oignon.rag augment "How to calculate diversity?"
    python -m opti_oignon.rag stats
    python -m opti_oignon.rag clear
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import get_config
from .indexer import DocumentIndexer
from .retriever import DocumentRetriever, format_results
from .augmenter import PromptAugmenter
from .embeddings import check_ollama_status


def cmd_index(args):
    """Index a folder."""
    indexer = DocumentIndexer()
    
    print(f"📁 Indexing: {args.path}")
    if args.force:
        print("   (Force re-indexing)")
    
    try:
        stats = indexer.index_directory(
            Path(args.path).expanduser(),
            recursive=not args.no_recursive,
            force=args.force
        )
        
        print(f"\n✅ Indexing complete:")
        print(f"   Files indexed: {stats['indexed_files']}")
        print(f"   Files skipped: {stats['skipped_files']}")
        print(f"   Chunks created: {stats['total_chunks']}")
        if stats.get('errors', 0) > 0:
            print(f"   ⚠️  Errors: {stats['errors']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_search(args):
    """Search documents."""
    retriever = DocumentRetriever()
    
    if retriever.count() == 0:
        print("⚠️  No documents indexed. Run 'index' first.")
        sys.exit(1)
    
    print(f"🔍 Searching: {args.query}\n")
    
    results = retriever.search(
        args.query,
        n_results=args.n,
        min_score=args.min_score
    )
    
    if not results:
        print("No results found.")
        return
    
    print(format_results(results, show_content=not args.brief))


def cmd_augment(args):
    """Augment a query with context."""
    augmenter = PromptAugmenter()
    
    print(f"💡 Augmenting query: {args.query}\n")
    
    result = augmenter.augment_smart(
        args.query,
        n_results=args.n
    )
    
    print(f"Context found: {result.has_context}")
    print(f"Sources: {result.sources_summary}")
    print(f"Context size: {result.total_context_chars} chars")
    print("\n" + "=" * 60)
    print("AUGMENTED PROMPT:")
    print("=" * 60)
    print(result.augmented_prompt)


def cmd_stats(args):
    """Show RAG statistics."""
    indexer = DocumentIndexer()
    stats = indexer.get_stats()
    
    print("\n📊 RAG Statistics")
    print("=" * 40)
    print(f"Total chunks:    {stats['total_chunks']}")
    print(f"Total files:     {stats['total_files']}")
    print(f"Collection:      {stats['collection_name']}")
    print(f"Storage:         {stats['storage_path']}")
    print(f"Embedding model: {stats['embedding_model']}")
    
    if stats.get('files_by_type'):
        print("\nFiles by type:")
        for ft, count in sorted(stats['files_by_type'].items()):
            print(f"  {ft}: {count}")
    
    # Ollama status
    print("\n🔌 Ollama Status")
    print("-" * 40)
    status = check_ollama_status()
    if status['ollama_running']:
        print(f"✅ Ollama running")
        print(f"   Embedding model: {'✅' if status['embedding_model_available'] else '❌'} {status['model_name']}")
    else:
        print(f"❌ Ollama not running")
        if status.get('error'):
            print(f"   Error: {status['error']}")


def cmd_clear(args):
    """Clear the index."""
    if not args.yes:
        confirm = input("⚠️  This will delete ALL indexed documents. Continue? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    
    indexer = DocumentIndexer()
    if indexer.clear_index():
        print("✅ Index cleared successfully.")
    else:
        print("❌ Error clearing index.")
        sys.exit(1)


def cmd_files(args):
    """List indexed files."""
    indexer = DocumentIndexer()
    files = indexer.list_indexed_files()
    
    if not files:
        print("No files indexed.")
        return
    
    print(f"\n📄 Indexed Files ({len(files)} total)")
    print("=" * 60)
    
    for f in files[:args.n]:
        print(f"  {f['type']:10} | {f['chunks']:3} chunks | {Path(f['path']).name}")
    
    if len(files) > args.n:
        print(f"\n  ... and {len(files) - args.n} more files")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Opti-Oignon RAG - Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m opti_oignon.rag index ~/Documents/code
  python -m opti_oignon.rag search "Shannon diversity"
  python -m opti_oignon.rag augment "How to do PCA in R?"
  python -m opti_oignon.rag stats
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index command
    p_index = subparsers.add_parser("index", help="Index a folder")
    p_index.add_argument("path", help="Folder to index")
    p_index.add_argument("-f", "--force", action="store_true", help="Force re-indexing")
    p_index.add_argument("--no-recursive", action="store_true", help="Don't index subfolders")
    p_index.set_defaults(func=cmd_index)
    
    # Search command
    p_search = subparsers.add_parser("search", help="Search documents")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", type=int, default=5, help="Number of results")
    p_search.add_argument("--min-score", type=float, default=0.3, help="Minimum score")
    p_search.add_argument("-b", "--brief", action="store_true", help="Brief output")
    p_search.set_defaults(func=cmd_search)
    
    # Augment command
    p_augment = subparsers.add_parser("augment", help="Augment a query with context")
    p_augment.add_argument("query", help="Query to augment")
    p_augment.add_argument("-n", type=int, default=3, help="Number of context chunks")
    p_augment.set_defaults(func=cmd_augment)
    
    # Stats command
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)
    
    # Clear command
    p_clear = subparsers.add_parser("clear", help="Clear the index")
    p_clear.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_clear.set_defaults(func=cmd_clear)
    
    # Files command
    p_files = subparsers.add_parser("files", help="List indexed files")
    p_files.add_argument("-n", type=int, default=20, help="Number of files to show")
    p_files.set_defaults(func=cmd_files)
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
