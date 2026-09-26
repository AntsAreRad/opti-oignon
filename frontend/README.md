# Opti-Oignon Frontend

SvelteKit-based web interface for the Opti-Oignon local LLM optimization suite.


## Architecture

The frontend communicates with the FastAPI backend (`opti_oignon/api/`) over REST.
It is a single-page application with client-side routing, streaming chat, and
real-time panel updates.

```
Backend (FastAPI :8000) <-- REST/SSE --> Frontend (SvelteKit :5173)
         |
    Ollama (local LLMs)
```


## Prerequisites

- Node.js >= 18
- npm >= 9
- Running Opti-Oignon API backend (`opti-oignon api` or `uvicorn opti_oignon.api.app:app`)


## Development Setup

```bash
# Install the dependencies package-lock.json pins; run it again after a
# pull that changes the lock (the launcher, `python -m opti_oignon`, does
# this by itself)
npm ci

# Start development server (hot reload)
npm run dev
# -> http://localhost:5173

# The backend API must be running on http://localhost:8001
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
  app.html              # HTML shell (the theme pre-render, sveltekit hooks)
  app.css               # Global styles: base, motion, focus ring, selection, forced colours

  styles/
    theme-day.css       # The day palette: its roles and its colour scheme
    theme-night.css     # The night palette
    theme-high-contrast.css  # The high contrast palette
    theme.css           # The derivation layer: every app token from the roles, once
    tokens.css          # Tokens without colour: fonts, text and space in rem, radii, motion
    density.css         # The text and space scales of each density

  lib/
    types.ts            # Shared TypeScript interfaces
    motion.ts           # The scroll behaviour the motion preference allows
    theme/
      apply.ts          # The theme path: stored choices and the system -> the root's state

    chat/
      requestFields.ts  # The ChatRequest fields a message sends, and their two builders
    switches/
      serverSwitch.ts   # A server setting shown as the server confirmed it, never as asked
    settings/
      catalog.ts        # Every settings section and group, and the old tab ids

    markdown/           # A reply's markdown as a closed node tree (no raw HTML)
      tree.ts           # marked's lexer (GFM, single newline = break) -> nodes, work bounded
      entities.ts       # Character references in text, from a fixed table
      collapse.ts       # A long reply collapsed by whole blocks, plain text by lines
      frame.ts          # At most one lex per animation frame; a slow lex is not repeated
      highlight.ts      # Two-role highlighter for code (keywords, defined names)

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
      ui.ts             # Sidebar visibility, whether the palette shown is dark, reduced motion
      chat.ts           # Streaming state, current response, abort controller
      chatOptions.ts    # Model, temperature, system prompt, preset selection
      notifications.ts  # Toast notification queue
      panels.ts         # Panel visibility (artifacts, code, memory, pipeline)

    components/
      chat/             # Chat interface (10 components)
        ChatMessage       # Message bubble: a reply's markdown, any other message as written, retry button
        markdown/         # Markdown, MarkdownNode, CodeBlock (label, Copy), MarkdownTable, PlainText, Caret
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

The frontend uses Tailwind CSS utility classes over design tokens (the
`--oo-*` custom properties). A component writes no hex and no named colour;
the `rgba()` literals some components still hold are a debt a ratchet only
lets fall.

Three palettes -- day, night and high contrast -- each declare the same
named roles (a ground, a surface, the text, the accent fill and its ink, a
rule, ...) in `src/styles/theme-*.css`, under `[data-oo-theme="<id>"]`.
`src/styles/theme.css` derives every app token from those roles, once, so a
subtree carrying `data-oo-theme="night"` previews that palette with no
colour of its own. Two tokens differ with the palette's scheme (the accent
fill's hover and the dialog's scrim); each palette gives them in a rule of
`theme.css` that names it. Tailwind's `surface-*` and `accent-*` utilities
resolve to the same tokens (`tailwind.config.js` holds no colour).

The accent fill is the ground of what carries text, with the on-accent ink
on it; a mark that carries no text (a progress bar, a legend dot) is drawn
in the mark token, and text on a status fill is the on-semantic ink.

One module, `src/lib/theme/apply.ts`, decides what `<html>` carries: the
palette ("Match system" by default: night, or day when the system asks for a
light scheme, or high contrast when it asks for more contrast), the `dark`
class, one density class, the root font size of the text size (92, 100, 109
or 118 percent: every rem follows it), the motion classes and the componion
switch. The inline script in `app.html` makes the same decisions before the
first paint, and when it cannot run the static night palette on `<html>`
stands. The preferences store stores an explicit choice and nothing else.

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

- One skip link, in the root layout, to the page's single `main-content` landmark
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
| `VITE_API_URL`        | same origin (empty)      | Backend API base URL. Unset, requests go to the page origin and the dev server proxies them; set it to `http://localhost:8001` to bypass the proxy. |


## API Communication

All API calls go through `lib/api/client.ts`, which provides:
- Automatic base URL configuration from `VITE_API_URL`
- JSON request/response handling
- Error normalization
- SSE streaming support for chat completions
