#!/usr/bin/env python3
"""
Basic usage example for Opti-Oignon.

This script demonstrates how to use Opti-Oignon programmatically
for simple query execution with automatic model routing.
"""

from opti_oignon.analyzer import TaskAnalyzer
from opti_oignon.router import ModelRouter
from opti_oignon.executor import Executor


def main():
    # Initialize components
    analyzer = TaskAnalyzer()
    router = ModelRouter()
    executor = Executor()
    
    # Example queries
    queries = [
        "Write a Python function to calculate the Fibonacci sequence",
        "Explain what a neural network is in simple terms",
        "Debug this code: for i in range(10) print(i)",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Step 1: Analyze the query
        analysis = analyzer.analyze(query)
        print(f"\nAnalysis:")
        print(f"   Task type: {analysis.task_type.value}")
        print(f"   Language: {analysis.language.value if analysis.language else 'N/A'}")
        print(f"   Complexity: {analysis.complexity.value}")
        print(f"   Suggested model type: {analysis.suggested_model_type}")
        
        # Step 2: Route to optimal model
        routing = router.route(analysis)
        print(f"\nRouting:")
        print(f"   Model: {routing.model}")
        print(f"   Temperature: {routing.temperature}")
        print(f"   Prompt variant: {routing.prompt_variant}")
        
        # Step 3: Execute query (simple mode, no streaming)
        print(f"\nResponse:")
        response = executor.execute_simple(
            question=query,
            model=routing.model,
            system_prompt=executor.get_system_prompt(routing.task_type),
            temperature=routing.temperature
        )
        # Truncate long responses for display
        print(response[:500] + "..." if len(response) > 500 else response)


def simple_example():
    """
    Even simpler example using the global instances.
    """
    from opti_oignon import analyze, router, executor
    
    query = "How do I read a CSV file in Python?"
    
    # One-liner style
    analysis = analyze(query)
    routing = router.route(analysis)
    
    # Stream response
    print("Streaming response:")
    for chunk in executor.execute(query, routing, refine=False):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
    # simple_example()  # Uncomment to try streaming
