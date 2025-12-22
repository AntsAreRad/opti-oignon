# Changelog

All notable changes to Opti-Oignon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-22

### Added
- **Context manager** - Context max length is now extracted from "ollama show" for each model and context can now accurately be monitored.

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
