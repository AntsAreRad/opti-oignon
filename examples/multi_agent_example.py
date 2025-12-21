#!/usr/bin/env python3
"""
Multi-Agent orchestration example for Opti-Oignon.

This script demonstrates how to use the multi-agent system
for complex tasks that benefit from multiple specialized models.
"""

from opti_oignon.agents import Orchestrator
from opti_oignon.agents.specialists import (
    PlannerAgent,
    CoderAgent,
    ReviewerAgent,
    create_planner_agent,
    create_coder_agent,
    create_reviewer_agent,
)


def orchestrator_example():
    """
    Example: Use the Orchestrator for automatic pipeline execution.
    
    The Orchestrator automatically detects the best pipeline
    and coordinates multiple agents.
    """
    print("Running Orchestrator Example")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # List available pipelines
    print("\nAvailable pipelines:")
    for pipeline in orchestrator.list_pipelines():
        print(f"   - {pipeline['id']}: {pipeline['description']}")
    
    # Define a task
    task = """
    Create a Python function to calculate biodiversity indices
    (Shannon and Simpson) from a species abundance matrix.
    """
    
    print(f"\nTask:\n{task}")
    
    # Auto-detect and run appropriate pipeline
    detected = orchestrator.detect_pipeline(task)
    print(f"\nDetected pipeline: {detected}")
    
    if detected:
        # Run the pipeline
        result = orchestrator.run_pipeline(
            pipeline_id=detected,
            query=task,
        )
        
        print("\n" + "="*60)
        print("Pipeline Results:")
        print("="*60)
        print(f"Status: {result.status}")
        print(f"Duration: {result.duration:.1f}s")
        print(f"\nFinal Output:\n{result.final_output[:1000]}...")
    else:
        print("No suitable pipeline detected, running single agent...")


def custom_pipeline():
    """
    Example: Create a custom pipeline with specific agents.
    
    Manually coordinate agents for fine-grained control.
    """
    print("Running Custom Pipeline")
    print("="*60)
    
    # Initialize individual agents using factory functions
    planner = create_planner_agent()
    coder = create_coder_agent()
    reviewer = create_reviewer_agent()
    
    task = "Write a Python function to parse CSV files with automatic type detection"
    
    print(f"\nTask: {task}")
    
    # Step 1: Planning
    print("\nStep 1: Planning...")
    plan = planner.create_plan(task)
    print(f"   Objective: {plan.objective}")
    print(f"   Steps identified: {len(plan.steps)}")
    for step in plan.steps[:5]:  # Show first 5 steps
        print(f"      {step.number}. {step.description[:60]}...")
    
    # Step 2: Coding
    print("\nStep 2: Coding...")
    code = coder.generate_code(
        request=task,
        language="python",
        context={"plan": str(plan)}
    )
    print(f"   Code generated: {len(code)} characters")
    print(f"   Preview:\n{code[:300]}...")
    
    # Step 3: Review
    print("\nStep 3: Reviewing...")
    review = reviewer.review_code(code)
    print(f"   Score: {review.score}/100")
    print(f"   Valid: {review.is_valid}")
    print(f"   Summary: {review.summary[:200]}...")
    
    # Step 4: Refinement (if needed)
    if not review.is_valid or review.score < 70:
        print("\nStep 4: Refining based on review...")
        refined_code = coder.generate_code(
            request=f"Improve this code based on feedback:\n\n{code}\n\nFeedback:\n{review.summary}",
            language="python",
        )
        final_code = refined_code
        print(f"   Refined code: {len(final_code)} characters")
    else:
        final_code = code
        print("\nCode passed review, no refinement needed")
    
    print("\n" + "="*60)
    print("Final Code:")
    print("="*60)
    print(final_code[:1500] + ("..." if len(final_code) > 1500 else ""))


def single_agent_examples():
    """
    Example: Use individual agents for specific tasks.
    """
    print("Running Single Agent Examples")
    print("="*60)
    
    # Coder agent
    print("\nCoderAgent Example:")
    coder = create_coder_agent()
    code = coder.generate_code(
        request="Write a function to calculate the mean of a list",
        language="python"
    )
    print(code)
    
    # Planner agent
    print("\nPlannerAgent Example:")
    planner = create_planner_agent()
    plan = planner.create_plan("Build a REST API for a todo list application")
    print(f"Plan with {len(plan.steps)} steps:")
    for step in plan.steps:
        print(f"   {step.number}. {step.description}")
    
    # Reviewer agent
    print("\nReviewerAgent Example:")
    reviewer = create_reviewer_agent()
    sample_code = '''
def add(a, b):
    return a + b
'''
    review = reviewer.review_code(sample_code)
    print(f"Score: {review.score}/100")
    print(f"Valid: {review.is_valid}")


def main():
    """Run examples."""
    print("\n" + "🧅 "*20)
    print("OPTI-OIGNON MULTI-AGENT EXAMPLES")
    print("🧅 "*20 + "\n")
    
    # Uncomment the example you want to run:
    
    # orchestrator_example()
    custom_pipeline()
    # single_agent_examples()


if __name__ == "__main__":
    main()
