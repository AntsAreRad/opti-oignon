# Opti-Oignon Frontend

SvelteKit-based web interface for the Opti-Oignon local LLM optimization suite.


## Architecture

The frontend communicates with the FastAPI backend (`opti_oignon/api/`) over REST.
It is a single-page application with client-side routing, streaming chat, and
real-time panel updates.

```
Backend (FastAPI :8000) <── REST/SSE ──> Frontend (SvelteKit :5173)
         │
    Ollama (local LLMs)
```


## Prerequisites

- Node.js >= 18
- npm >= 9
- Running Opti-Oignon API backend (`opti-oignon api` or `uvicorn opti_oignon.api.app:app`)


## Development Setup

```bash
# Install dependencies
npm install

# Start development server (hot reload)
npm run dev
# -> http://localhost:5173

# The backend API must be running on http://localhost:8000
# Start it with: opti-oignon api
```


## Build for Production

```bash
npm run build
npm run preview   # preview the production build locally
```


## Project Structure

```
src/
  app.html              # HTML shell (skip-to-content, sveltekit hooks)
  app.css               # Global styles, animations, theme system

  lib/
    types.ts            # Shared TypeScript interfaces

    api/                # REST client modules (15 modules)
      client.ts         # Base HTTP client (fetch wrapper, error handling)
      conversations.ts  # Conversation CRUD
      models.ts         # Ollama model listing
      chat.ts           # Chat completion + SSE streaming
      presets.ts        # Preset management
      artifacts.ts      # Artifact CRUD
      code.ts           # Sandboxed code execution
      files.ts          # File upload
      memory.ts         # Cross-conversation memory
      search.ts         # Web search integration
      pipelines.ts      # Pipeline management + execution
      settings.ts       # User settings
      health.ts         # System health + Ollama status
      cache.ts          # Response cache management
      export.ts         # Conversation export (Markdown/JSON/HTML)

    stores/             # Svelte stores (6 stores)
      conversations.ts  # Conversation list, selection, messages, loading state
      ui.ts             # Sidebar visibility, theme, global UI state
      chat.ts           # Streaming state, current response, abort controller
      chatOptions.ts    # Model, temperature, system prompt, preset selection
      notifications.ts  # Toast notification queue
      panels.ts         # Panel visibility (artifacts, code, memory, pipeline)

    components/
      chat/             # Chat interface (10 components)
        ChatMessage       # Message bubble with role icon, markdown, retry button
        ChatInput         # Textarea with send/cancel, file attach, keyboard submit
        FileUpload        # Drag-and-drop + click file upload
        ContextBar        # Active model/preset/temperature display
        StreamingIndicator  # Animated dots during generation
        ModelSelector     # Model dropdown (from Ollama)
        PresetSelector    # Preset dropdown with icons
        SearchResults     # Web search result cards
        ExportDialog      # Modal: format selector, preview, download/copy
        MessageSkeleton   # Pulsing loading placeholder for messages

      sidebar/          # Conversation sidebar (4 components)
        ConversationList  # Scrollable list with search, skeleton loading
        ConversationItem  # Single conversation row (rename, delete, export)
        NewConversationButton  # Create new conversation
        ConversationSkeleton   # Pulsing loading placeholder for sidebar

      layout/           # Layout shell (2 components)
        AppShell          # Main layout: sidebar + content + panel
        Sidebar           # Sidebar wrapper with toggle, theme button

      panels/           # Feature panels (6 components)
        ArtifactPanel     # Artifact viewer with version history
        CodePanel         # Code execution with output display
        MemoryPanel       # Cross-conversation memory facts
        PanelToggle       # Panel open/close toggle buttons
        PipelinePanel     # Pipeline status and step display
        PipelineStepEditor  # Edit individual pipeline steps

      settings/         # Settings (1 component)
        PresetManager     # Create, edit, delete, reorder presets

      health/           # System health (2 components)
        HealthDashboard   # Ollama status, model list, system info
        CacheManager      # Cache stats, clear cache

      ui/               # Shared UI (3 components)
        Toast             # Notification toasts (aria-live)
        KeyboardShortcuts # Global shortcuts + help overlay modal
        ErrorBoundary     # Error wrapper with retry button

  routes/
    +layout.svelte      # Root layout: theme init, shortcuts, toasts
    +layout.ts          # SvelteKit layout config (SSR disabled)
    +page.svelte        # Home redirect to /chat

    chat/
      +layout.svelte    # Chat layout: AppShell, header, export dialog
      +page.svelte      # Empty state (no conversation selected)
      [id]/+page.svelte # Active conversation: messages, input, panels

    settings/
      +layout.svelte    # Settings layout
      +page.svelte      # Settings page: tabs for presets, general

    health/
      +layout.svelte    # Health layout
      +page.svelte      # Health dashboard + cache manager
```


## Component Count

- 28 components total
- 15 API modules
- 6 stores
- 62+ files


## Styling

The frontend uses Tailwind CSS utility classes with a custom dark/light theme system.

Theme switching is handled via the `dark` class on `<html>`. Custom CSS
variables and overrides are defined in `app.css`.

CSS animations (all in `app.css`):
- `message-in`: fade + slide-up for new messages (250ms)
- `panel-slide`: slide-from-right for panel open (200ms)
- `fade-in`: generic fade for modals (200ms)
- `sidebar-slide`: slide-from-left for sidebar (200ms)
- `skeleton-pulse`: pulsing placeholder animation (1.5s infinite)


## Keyboard Shortcuts

| Shortcut         | Action                  |
|------------------|-------------------------|
| `Ctrl+N`         | New conversation        |
| `Ctrl+Shift+E`   | Export conversation     |
| `Ctrl+,`         | Open settings           |
| `Ctrl+K`         | Focus search            |
| `?`              | Show shortcuts help     |
| `Escape`         | Close modal / panel     |


## Accessibility

- Skip-to-content link in `app.html`
- `aria-live="polite"` on toast notifications
- `role="dialog"` + `aria-modal` on all modals
- Focus trap (Tab cycling) in ExportDialog and KeyboardShortcuts
- Focus management: return focus to trigger on modal close
- `aria-labels` on all icon-only buttons
- `role="log"` on the chat message area
- Minimum viewport width: 320px


## Environment Variables

| Variable              | Default                  | Description            |
|-----------------------|--------------------------|------------------------|
| `VITE_API_URL`        | `http://localhost:8000`  | Backend API base URL   |


## API Communication

All API calls go through `lib/api/client.ts`, which provides:
- Automatic base URL configuration from `VITE_API_URL`
- JSON request/response handling
- Error normalization
- SSE streaming support for chat completions
