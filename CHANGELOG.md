# Changelog

All notable changes to Opti-Oignon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-12-25

### Added

- **Pipeline Manager** - New UI tab for managing multi-agent pipelines
  - View all pipelines (builtin + custom) in a searchable table
  - Create custom pipelines with visual step-by-step editor (up to 10 steps)
  - Modify and delete custom pipelines (builtin are read-only)
  - Duplicate any pipeline for customization
  - Import/Export pipelines in YAML format
  - LLM-powered prompt generation for pipeline steps
  - Keywords for automatic pipeline detection with weighted scoring
  - Move Up/Down buttons (↑↓) to reorder pipeline steps

- **Model Override per Step** - Pipeline steps can now use a specific model
  - Added "Model" dropdown in each pipeline step
  - Allows overriding the agent's default model
  - Model override is saved in `pipelines_custom.yaml`

- **Keepalive Mechanism** - Prevents Gradio timeouts during long operations
  - Threading + queue-based communication for responsive UI
  - Applied to both single queries and multi-agent pipelines
  - Progress indicators during model loading

- **Dynamic Step Addition** - Cleaner pipeline creation UX
  - Start with only 2 visible steps
  - "➕ Add Step" button reveals additional steps (up to 10)
  - All steps use collapsible Accordion layout

- **Template Loading** - Templates load their content into the prompt field
  - Selecting a template fills the System Prompt with template content
  - Auto-switch to Custom mode when editing a template prompt

### Fixed

- **Custom Pipelines Not Visible in Chat** - Critical persistence bug fixed
  - Custom pipelines now properly appear in the "Select Mode" dropdown
  - Dropdown automatically refreshes after creating/updating/deleting pipelines
  - Pipeline changes in Pipelines tab sync to Chat tab immediately

- **Custom Presets Not Visible in Chat** - Same fix applied to presets
  - Custom presets now properly appear in the "Preset" dropdown
  - Preset changes sync between tabs immediately

- **Chat Timeout During Model Loading** - Gradio timeout fix
  - Added keepalive mechanism using threading
  - Prevents connection timeout when model is loading into memory

- **Template Loading Errors** - Fixed `'str' object has no attribute 'get'`
  - Templates in config.yaml are now properly handled as strings

- **Empty Template Dropdown Error** - Fixed "Value not in list of choices"
  - Added `allow_custom_value=True` to template dropdowns
  - Templates used in pipelines are auto-added to dropdown choices

- **Ollama Connection** - Fixed model name resolution
  - Now returns full model name with tag (e.g., `qwen3:32b`)
  - Better error messages for connection issues

### Changed

- **Pipeline Manager UI** - Major UX improvements
  - Replaced JSON step editor with visual "Add Step" interface
  - 5→10 step blocks with Name, Agent, Model, Prompt Type, Prompt, Description
  - Each step has a "Generate" button for LLM-powered prompt generation
  - Collapsible accordions (Step 1-2 open by default)

- **Multi-Agent Tab Removed** - Consolidated into Pipelines tab
  - Enable/Disable toggle moved to top of Pipelines tab
  - Less UI clutter, more intuitive navigation

- **Layout Balance** - Better column proportions in Pipelines tab
  - Available Pipelines: 40% width
  - Create/Edit Pipeline: 60% width

### Technical

- **New modules:**
  - `pipeline_manager.py`: Pipeline and PipelineStep dataclasses, PipelineManager with CRUD
  - `dynamic_pipeline_ui.py`: UI integration for dynamic pipeline planning

- **New data structures:**
  - `PipelineStep.model`: Optional model override field
  - `STEP_FIELDS`: Increased from 7 to 8 fields per step

- **New functions:**
  - `get_pipeline_dropdown_update()`: Returns `gr.update(choices=...)`
  - `get_preset_choices()` / `get_preset_dropdown_update()`: Preset dropdown refresh
  - `get_available_models_list()`: Populates model dropdowns
  - `swap_steps()`: Exchanges content between adjacent steps
  - `get_template_content()`: Loads template from orchestrator config

- **Modified functions:**
  - `executor.execute()`: Rewritten with threading + queue for keepalive
  - `refresh_multi_agent_stats()`: Now returns both status and dropdown update
  - Pipeline/Preset CRUD functions: Return 3 values (status, table, dropdown_update)

- **Storage:**
  - Custom pipelines: `opti_oignon/data/pipelines_custom.yaml`
  - Orchestrator loads both builtin and custom pipelines

---

## [1.1.0] - 2025-12-22

### Added
- **Context Manager** - Context max length is now extracted from "ollama show" for each model and context can now accurately be monitored.
- **Multi-agent history metadata** - Pipeline ID, step count, and step summaries are now saved in history.

### Fixed
- **Document upload** - Files uploaded via the Gradio interface are now properly sent to the model.
- **Multi-agent history** - Conversations using multi-agent pipelines are now saved to history.
- **Context indicator** - Now correctly includes uploaded files, works with presets and auto-routing.

### Changed
- **Default UI settings** - Multi-Agent and RAG are now disabled by default.

---

## [1.0.0] - 2025-12-21

### Added
- Initial release
- Automatic task detection and model routing
- RAG document enrichment
- Multi-agent orchestration with pipelines
- Preset system with keyword-based auto-detection
- Conversation history with search and export
- Gradio web interface with dark mode
