#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) example for Opti-Oignon.

This script demonstrates how to:
1. Index documents into the vector database
2. Search for relevant content
3. Use retrieved context to enhance LLM responses
"""

from pathlib import Path
from opti_oignon.rag import (
    DocumentIndexer,
    DocumentRetriever,
    PromptAugmenter,
    SearchResult,
)
from opti_oignon.executor import Executor


def main():
    """Main RAG demonstration."""
    
    # Initialize RAG components
    print("Initializing RAG components...")
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    augmenter = PromptAugmenter()
    executor = Executor()
    
    # ===================
    # Step 1: Check indexed documents
    # ===================
    print("\nChecking indexed documents...")
    total_chunks = retriever.count()
    file_types = retriever.get_file_types()
    print(f"   Total chunks indexed: {total_chunks}")
    print(f"   File types: {', '.join(file_types) if file_types else 'none'}")
    
    # Index a directory if you have documents
    docs_path = Path("./my_documents")
    if docs_path.exists():
        print(f"\nIndexing {docs_path}...")
        index_stats = indexer.index_directory(str(docs_path), recursive=True)
        print(f"   Files processed: {index_stats.get('indexed_files', 0)}")
        print(f"   Chunks created: {index_stats.get('total_chunks', 0)}")
    else:
        print(f"   (No {docs_path} directory found - using existing index)")
    
    # ===================
    # Step 2: Search for relevant content
    # ===================
    print("\n" + "="*60)
    print("Searching indexed content...")
    print("="*60)
    
    query = "How do I calculate Shannon diversity index?"
    results = retriever.search(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant chunks:")
    
    for i, result in enumerate(results, 1):
        print(f"\n   [{i}] Score: {result.score:.3f}")
        print(f"       Source: {result.source}")
        print(f"       Type: {result.file_type}")
        preview = result.content[:100].replace('\n', ' ')
        print(f"       Content: {preview}...")
    
    # ===================
    # Step 3: Augmented generation
    # ===================
    print("\n" + "="*60)
    print("Generating augmented response...")
    print("="*60)
    
    # Create enhanced prompt with retrieved context
    augmented = augmenter.augment(
        query=query,
        n_results=5,
        min_score=0.3,
        template="code"  # Use code template for technical queries
    )
    
    print(f"\nOriginal query: {augmented.original_query}")
    print(f"Context chunks used: {len(augmented.context_chunks)}")
    print(f"Total context chars: {augmented.total_context_chars}")
    
    # Show the augmented prompt (truncated)
    print(f"\nAugmented prompt preview:")
    print(augmented.augmented_prompt[:500] + "...")
    
    # Execute with the enhanced prompt
    print("\nModel response (with RAG context):")
    response = executor.execute_simple(
        question=augmented.augmented_prompt,
        model="qwen3-coder:30b",
        system_prompt="You are a helpful assistant with expertise in R and bioinformatics.",
        temperature=0.3
    )
    print(response)


def index_folder_example():
    """
    Example: Index a specific folder of documents.
    """
    print("Indexing folder example")
    print("="*60)
    
    indexer = DocumentIndexer()
    
    # Index a folder
    folder_path = "/path/to/your/documents"
    
    if Path(folder_path).exists():
        stats = indexer.index_directory(
            folder_path,
            recursive=True,  # Include subdirectories
            force_reindex=False  # Only index new/modified files
        )
        
        print(f"Indexed {stats.get('indexed_files', 0)} files")
        print(f"Created {stats.get('total_chunks', 0)} chunks")
        print(f"Skipped {stats.get('skipped_files', 0)} unchanged files")
    else:
        print(f"Folder not found: {folder_path}")


def search_example():
    """
    Example: Various search operations.
    """
    print("Search examples")
    print("="*60)
    
    retriever = DocumentRetriever()
    
    # Basic search
    results = retriever.search("diversity index", n_results=5)
    print(f"\nBasic search: {len(results)} results")
    
    # Search with file type filter
    results = retriever.search(
        "ggplot visualization",
        n_results=5,
        file_types=[".r", ".R"]  # Only R files
    )
    print(f"R files only: {len(results)} results")
    
    # Search with minimum score
    results = retriever.search(
        "Shannon Simpson",
        n_results=10,
        min_score=0.5  # Only high-relevance results
    )
    print(f"High relevance (>0.5): {len(results)} results")


if __name__ == "__main__":
    main()
    # index_folder_example()
    # search_example()
