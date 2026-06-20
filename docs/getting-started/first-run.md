# First Run

## Onboarding overlay

When you open Opti-Oignon for the first time, the **onboarding overlay**
appears automatically. It performs three steps:

1. **Model scan** -- detects all models available in your local Ollama
   instance and displays their sizes and capabilities
2. **Preset recommendation** -- based on your available models and system
   resources, recommends one of three presets:
   - **Minimal** -- lightweight, single small model, no consensus or
     cascading; best for machines with limited RAM
   - **Balanced** -- two models with smart routing, chain-of-thought
     enabled; good default for most setups
   - **Power** -- all available models, consensus voting, cascading
     inference, speculative generation; requires 16+ GB RAM
3. **One-click apply** -- selecting a preset configures all YAML files
   at once (routing, pipelines, security, plugins)

You can change your preset later in **Settings > Quick tab**.


## First conversation

After onboarding:

1. Type a message in the chat input and press `Ctrl+Enter` (or click Send)
2. Opti-Oignon routes your query to the best available model
3. The response streams in real time via WebSocket

The sidebar shows your conversation history. You can start a new chat
with `Ctrl+N`.


## Settings overview

Open settings with `Ctrl+,` or the gear icon. The settings page has two
tabs:

- **Quick** -- preset selection, default model, theme toggle
- **Advanced** -- per-pipeline configuration, security settings, plugin
  management, RAG configuration, benchmark dashboard


## Security on first run

By default, Opti-Oignon starts in **Daily mode** with standard security.
No authentication is required for local-only access.

To enable full security (recommended for any networked use):

1. Go to **Settings > Advanced > Security**
2. Create an admin account with a strong password
3. Optionally enable 2FA (TOTP or WebAuthn/FIDO2)
4. Consider enabling **Bulbe mode** for maximum security
   (see [Bulbe Mode](../security/bulbe-mode.md))


## What next

- [Chat features](../user-guide/chat.md) -- explore pipelines, consensus,
  coding agent
- [RAG projects](../user-guide/rag.md) -- ingest documents and query them
- [Security guide](../security/overview.md) -- understand the defense layers
- [CLI reference](../cli-reference.md) -- use `oo` from your terminal
