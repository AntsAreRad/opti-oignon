# Pipeline Manager - User Guide

## Overview

The Pipeline Manager is a feature introduced in Opti-Oignon v1.2.0 that allows you to create, modify, and manage custom multi-agent pipelines directly from the user interface.

## Key Concepts

### Pipeline Types

- **Builtin**: Predefined pipelines in `agents/config.yaml`. Read-only, but can be duplicated.
- **Custom**: User-created pipelines. Stored in `data/pipelines_custom.yaml`.

### Pipeline Structure

```yaml
id: my_pipeline
name: "My Custom Pipeline"
description: "Pipeline description"
pattern: chain  # chain, verifier, decomposition, iterative
steps:
  - name: "Analysis"
    agent: "reviewer"
    prompt_template: "error_analysis"  # OR system_prompt
    description: "Analyze the problem"
  - name: "Solution"
    agent: "coder"
    system_prompt: "You are an expert..."
    description: "Propose a solution"
keywords:
  - "debug"
  - "error"
detection_weight: 0.7  # 0.0 - 1.0
```

### Available Agents

| Agent | Description | Default Model |
|-------|-------------|---------------|
| `coder` | Code generation | qwen3-coder:30b |
| `reviewer` | Review and verification | deepseek-r1:32b |
| `explainer` | Explanations | qwen3:32b |
| `planner` | Planning | deepseek-r1:32b |
| `writer` | Scientific writing | qwen3:32b |
| `fast_planner` | Quick planning | qwen3:32b |
| `fast_understanding` | Quick comprehension | nemotron-3-nano:30b |

### Orchestration Patterns

- **chain**: Simple sequential execution (Step 1 > Step 2 > Step 3)
- **verifier**: With final verification step
- **decomposition**: Breaks down the problem into subtasks
- **iterative**: Iterates until satisfaction

## User Interface

### Pipelines Tab

The **Pipelines** tab is accessible in the application settings.

#### Pipeline List

- Table displaying all pipelines (builtin + custom)
- Columns: Type, ID, Name, Steps Count, Pattern, Weight, Keywords
- Clicking a row loads its details into the form

#### Edit Form

1. **General Information**
   - ID: Unique identifier (no spaces)
   - Name: Display name
   - Description: Pipeline description
   - Pattern: Orchestration type

2. **Keywords**
   - Comma-separated list of keywords
   - Used for automatic pipeline detection
   - Detection weight (0.0-1.0): Priority when conflicts occur

3. **Steps**
   - Visual editor to define steps
   - Each step: name, agent, model (optional), prompt type, prompt, description
   - Up to 10 steps per pipeline
   - Use up/down arrows to reorder steps

#### Actions

- **Create New**: Create a new custom pipeline
- **Update Selected**: Update a custom pipeline (not builtin)
- **Delete**: Delete a custom pipeline
- **Duplicate**: Duplicate any pipeline

#### Import/Export

- **Export All**: Export all pipelines to YAML
- **Import**: Import pipelines from a YAML file

### LLM Prompt Generation

The "Generate Step Prompt" accordion allows automatic generation of optimized system prompts for a step:

1. Enter the step name
2. Enter the description of what the step should do
3. Enter the pipeline context (global objective)
4. Select the agent type
5. Click "Generate Prompt"
6. The generated prompt is inserted into the step

## Files

### Storage

- **Builtin**: `opti_oignon/agents/config.yaml` (read-only)
- **Custom**: `opti_oignon/data/pipelines_custom.yaml` (editable)

### Export Format

```yaml
pipelines:
  my_custom_pipeline:
    name: "My Pipeline"
    description: "..."
    pattern: "chain"
    steps:
      - name: "Step 1"
        agent: "coder"
        prompt_template: "direct"
        description: "..."
    auto_detect:
      keywords:
        - "keyword1"
        - "keyword2"
    detection_weight: 0.5
    created_at: "2025-12-23T10:00:00"
    is_builtin: false
```

## Python API

```python
from opti_oignon.pipeline_manager import (
    get_pipeline_manager,
    Pipeline,
    PipelineStep,
)

# Get the manager
pm = get_pipeline_manager()

# List all pipelines
for pipeline in pm.list_all():
    print(f"{pipeline.id}: {pipeline.name}")

# Create a new pipeline
new_pipeline = Pipeline(
    id="my_new_pipeline",
    name="My New Pipeline",
    description="Description",
    pattern="chain",
    steps=[
        PipelineStep(
            name="Analysis",
            agent="reviewer",
            system_prompt="You are an expert analyst...",
            description="Analyze the problem",
        ),
    ],
    keywords=["analysis", "debug"],
    detection_weight=0.7,
)
pm.create(new_pipeline)

# Get a pipeline
pipeline = pm.get("data_analysis")

# Delete a custom pipeline
pm.delete("my_old_pipeline")

# Export
yaml_content = pm.export_all()

# Import
imported_ids = pm.import_from_yaml(yaml_content)
```

## Important Notes

1. **Builtin pipelines** cannot be modified directly. Duplicate them first.

2. **Validation**: Each step must have either `prompt_template` (ID of an existing template) or `system_prompt` (custom prompt).

3. **Keywords**: Automatic detection uses the weighted score `matches x detection_weight`.

4. **Saving**: Modifications are automatically saved to `data/pipelines_custom.yaml`.

## Troubleshooting

### Pipeline not showing

- Check the file `data/pipelines_custom.yaml`
- Click "Reload Config" in the UI
- Verify the pipeline has at least one step

### Creation error

- ID must be unique
- ID must start with a letter
- At least one step is required

### Generated prompt is empty

- Verify Ollama is running
- Verify the model `qwen3:32b` is available
- Check network connection to Ollama
