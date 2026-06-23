<!--
  ChatControlBar.svelte
  Horizontal control bar above the input area.
  Provides model/preset selectors and feature toggles with responsive labels.
  Labels collapse to icon-only on screens < 640px (sm breakpoint).
  All active toggles use a borderless tobacco tint (S93 v4e palette).
  S87: responsive labels, unified style, ddgs availability check,
       model family grouping with parameter badges.
  S132: Mobile responsive — horizontal scroll overflow, touch-friendly min-height.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		selectedModel,
		selectedPreset,
		availableModels,
		availablePresets,
		thinkingEnabled,
		webSearchEnabled,
		cacheEnabled,
		cascadingEnabled,
		promptEnhanceEnabled,
		humanizeEnabled,
		quickSandboxEnabled,
		chatCodingEnabled,
		loadOptions,
	} from '$lib/stores/chatOptions';
	import { activeConversationId } from '$lib/stores/conversations';

	let loaded = false;

	// S87: Track whether duckduckgo-search is installed
	let ddgsAvailable = true;

	// S117: Track whether quick sandbox is available
	let qsAvailable = false;

	// S118: Track whether chat coding agent is available
	let ccAvailable = false;

	// S131: Conversation wipe
	let wipeAvailable = false;
	let wipeBusy = false;

	// S87: Group models by family for the dropdown
	interface ModelGroup {
		family: string;
		models: { name: string; paramBadge: string; mtpCapable: boolean }[];
	}

	let modelGroups: ModelGroup[] = [];

	$: {
		const grouped = new Map<string, { name: string; paramBadge: string; mtpCapable: boolean }[]>();
		for (const m of $availableModels) {
			const family = parseFamily(m.name);
			const badge = parseParamBadge(m.name, m.parameter_size);
			if (!grouped.has(family)) grouped.set(family, []);
			grouped.get(family)!.push({ name: m.name, paramBadge: badge, mtpCapable: m.mtp_capable ?? false });
		}
		modelGroups = Array.from(grouped.entries())
			.sort(([a], [b]) => a.localeCompare(b))
			.map(([family, models]) => ({ family, models }));
	}

	function parseFamily(name: string): string {
		// Extract family from model name (e.g. "qwen3-coder:30b" -> "Qwen")
		const lower = name.toLowerCase().split(':')[0].split('-')[0];
		const families: Record<string, string> = {
			qwen: 'Qwen', qwen2: 'Qwen', qwen3: 'Qwen',
			llama: 'Llama', llama2: 'Llama', llama3: 'Llama',
			gemma: 'Gemma', gemma2: 'Gemma', gemma3: 'Gemma',
			deepseek: 'DeepSeek', phi: 'Phi', phi3: 'Phi', phi4: 'Phi',
			mistral: 'Mistral', mixtral: 'Mistral',
			codellama: 'CodeLlama', codegemma: 'Gemma',
			command: 'Command', starcoder: 'StarCoder',
			yi: 'Yi', vicuna: 'Vicuna', orca: 'Orca',
			granite: 'Granite', falcon: 'Falcon',
			nomic: 'Nomic', mxbai: 'Mxbai',
		};
		return families[lower] || lower.charAt(0).toUpperCase() + lower.slice(1);
	}

	function parseParamBadge(name: string, parameterSize: string | null): string {
		// Try parameter_size from API first
		if (parameterSize) return parameterSize;
		// Fallback: extract from name (e.g. "qwen3:32b" -> "32B")
		const match = name.match(/(\d+\.?\d*)[bB]/);
		return match ? match[1] + 'B' : '';
	}

	// Unified active toggle style (borderless, tobacco tint, S93 v4e palette)
	const activeStyle = 'background-color: var(--oo-tobacco-bg); color: var(--oo-tobacco); border: 1px solid var(--oo-tobacco-bg);';
	const inactiveStyle = 'background-color: var(--oo-bg-surface); color: var(--oo-fg-muted); border: 1px solid var(--oo-bg-surface);';
	const disabledStyle = 'background-color: var(--oo-bg-surface); color: var(--oo-fg-muted); border: 1px solid var(--oo-bg-surface); opacity: 0.5; cursor: not-allowed;';

	onMount(async () => {
		if ($availableModels.length === 0) {
			await loadOptions();
		}
		// S68: Load initial cache status
		try {
			const resp = await fetch('/api/cache/s68/status');
			if (resp.ok) {
				const data = await resp.json();
				cacheEnabled.set(data.enabled || false);
			}
		} catch { /* best-effort: ignore if endpoint unavailable */ }
		// S69: Load initial cascading status
		try {
			const resp = await fetch('/api/cascading/status');
			if (resp.ok) {
				const data = await resp.json();
				cascadingEnabled.set(data.enabled || false);
			}
		} catch { /* best-effort: ignore if endpoint unavailable */ }
		// S86: Load initial humanizer status
		try {
			const resp = await fetch('/api/humanizer/config');
			if (resp.ok) {
				const data = await resp.json();
				humanizeEnabled.set(data.enabled || false);
			}
		} catch { /* best-effort: ignore if endpoint unavailable */ }
		// S87: Check if duckduckgo-search is available
		try {
			const resp = await fetch('/api/search/config');
			if (resp.ok) {
				const data = await resp.json();
				ddgsAvailable = data.ddgs_available ?? true;
			}
		} catch {
			// If search config endpoint unreachable, assume available
		}
		// S117: Load quick sandbox status
		try {
			const resp = await fetch('/api/sandbox/quick/status');
			if (resp.ok) {
				const data = await resp.json();
				qsAvailable = data.available ?? false;
				quickSandboxEnabled.set(data.enabled ?? false);
			}
		} catch {
			qsAvailable = false;
		}
		// S118: Load chat coding agent status
		try {
			const resp = await fetch('/api/chat/coding/status');
			if (resp.ok) {
				const data = await resp.json();
				ccAvailable = data.available ?? false;
				chatCodingEnabled.set(data.enabled ?? false);
			}
		} catch {
			ccAvailable = false;
		}
		// S131: Check conversation wipe availability
		try {
			const resp = await fetch('/api/security/hardening/status', { credentials: 'include' });
			if (resp.ok) {
				const data = await resp.json();
				wipeAvailable = data.conversation_wipe?.available ?? false;
			}
		} catch {
			wipeAvailable = false;
		}
		loaded = true;
	});

	function toggleThinking() {
		thinkingEnabled.update((v) => !v);
	}

	function toggleSearch() {
		if (!ddgsAvailable) return;
		webSearchEnabled.update((v) => !v);
	}

	async function toggleCache() {
		try {
			const resp = await fetch('/api/cache/s68/toggle', { method: 'POST' });
			if (resp.ok) {
				const data = await resp.json();
				cacheEnabled.set(data.enabled || false);
			} else {
				cacheEnabled.update((v) => !v);
			}
		} catch {
			cacheEnabled.update((v) => !v);
		}
	}

	async function toggleCascading() {
		const newVal = !$cascadingEnabled;
		try {
			const resp = await fetch('/api/cascading/config', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ enabled: newVal }),
			});
			if (resp.ok) {
				const data = await resp.json();
				cascadingEnabled.set(data.enabled || false);
			} else {
				cascadingEnabled.set(newVal);
			}
		} catch {
			cascadingEnabled.set(newVal);
		}
	}

	function togglePromptEnhance() {
		promptEnhanceEnabled.update((v) => !v);
	}

	async function toggleHumanize() {
		const newVal = !$humanizeEnabled;
		try {
			const resp = await fetch('/api/humanizer/config', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ enabled: newVal }),
			});
			if (resp.ok) {
				const data = await resp.json();
				humanizeEnabled.set(data.enabled || false);
			} else {
				humanizeEnabled.set(newVal);
			}
		} catch {
			humanizeEnabled.set(newVal);
		}
	}

	async function toggleQuickSandbox() {
		if (!qsAvailable) return;
		// S118: Mutual exclusion — disable Code Agent when toggling Sandbox on
		if (!$quickSandboxEnabled && $chatCodingEnabled) {
			chatCodingEnabled.set(false);
		}
		const newVal = !$quickSandboxEnabled;
		try {
			const resp = await fetch('/api/sandbox/quick/toggle', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ enabled: newVal }),
			});
			if (resp.ok) {
				const data = await resp.json();
				quickSandboxEnabled.set(data.enabled || false);
			} else {
				quickSandboxEnabled.set(newVal);
			}
		} catch {
			quickSandboxEnabled.set(newVal);
		}
	}

	async function toggleChatCoding() {
		if (!ccAvailable) return;
		// S118: Mutual exclusion — when Code Agent is ON, Sandbox is implicitly ON
		// When toggling Code Agent on, disable standalone Sandbox toggle
		const newVal = !$chatCodingEnabled;
		if (newVal && $quickSandboxEnabled) {
			quickSandboxEnabled.set(false);
		}
		try {
			const resp = await fetch('/api/chat/coding/toggle', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ enabled: newVal }),
			});
			if (resp.ok) {
				const data = await resp.json();
				chatCodingEnabled.set(data.enabled || false);
			} else {
				chatCodingEnabled.set(newVal);
			}
		} catch {
			chatCodingEnabled.set(newVal);
		}
	}

	// S87: Compute search toggle tooltip
	$: searchTooltip = ddgsAvailable
		? 'Toggle web search (DuckDuckGo)'
		: 'Install duckduckgo-search to enable (pip install duckduckgo-search)';

	// S131: Wipe current conversation
	async function handleWipeConversation() {
		const convId = $activeConversationId;
		if (!convId || wipeBusy) return;
		wipeBusy = true;
		try {
			await fetch(`/api/security/conversation-wipe/${encodeURIComponent(convId)}`, {
				method: 'POST',
				credentials: 'include',
			});
		} catch { /* best-effort: ignore if endpoint unavailable */ }
		wipeBusy = false;
	}
</script>

<!-- S132: Horizontal scroll on mobile, no-wrap to prevent overflow stacking -->
<div class="flex items-center gap-2 px-1 py-1.5 overflow-x-auto touch-scroll-x mobile-hide-scrollbar"
	style="min-height: 36px; -ms-overflow-style: none; scrollbar-width: none;">
	<!-- Model selector with family grouping and param badges (S87) -->
	<div class="flex items-center gap-1 shrink-0">
		<select
			bind:value={$selectedModel}
			class="text-xs rounded-lg px-2 py-1 outline-none cursor-pointer appearance-none pr-6"
			style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
				border: 1px solid var(--oo-input-bd);"
			title="Model"
			aria-label="Select model"
		>
			<option value={null}>Auto</option>
			{#if loaded}
				{#each modelGroups as group}
					<optgroup label={group.family}>
						{#each group.models as model}
							<option value={model.name}>
								{model.name}{model.paramBadge ? ` (${model.paramBadge})` : ''}{model.mtpCapable ? ' [MTP]' : ''}
							</option>
						{/each}
					</optgroup>
				{/each}
			{/if}
		</select>
	</div>

	<!-- Preset selector -->
	<div class="flex items-center gap-1 shrink-0">
		<select
			bind:value={$selectedPreset}
			class="text-xs rounded-lg px-2 py-1 outline-none cursor-pointer appearance-none pr-6"
			style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
				border: 1px solid var(--oo-input-bd);"
			title="Preset"
			aria-label="Select preset"
		>
			<option value={null}>Auto</option>
			{#if loaded}
				{#each $availablePresets as preset}
					<option value={preset.id}>{preset.name}</option>
				{/each}
			{/if}
		</select>
	</div>

	<!-- Visual separator -->
	<div class="w-px h-4 hidden sm:block" style="background-color: var(--oo-bd-default);" />

	<!-- Toggle Think -->
	<button
		on:click={toggleThinking}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{$thinkingEnabled ? activeStyle : inactiveStyle}"
		title="Toggle thinking mode (chain-of-thought reasoning)"
		aria-label="Toggle thinking mode"
		aria-pressed={$thinkingEnabled}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<path d="M9.5 2a5.5 5.5 0 00-3.36 9.86A3.5 3.5 0 007 18.5h1.5" />
			<path d="M14.5 2a5.5 5.5 0 013.36 9.86A3.5 3.5 0 0117 18.5h-1.5" />
			<path d="M8.5 18.5V22" />
			<path d="M15.5 18.5V22" />
			<path d="M12 2v4" />
			<path d="M12 10v4" />
		</svg>
		<span class="hidden sm:inline">Think</span>
	</button>

	<!-- Toggle Search (disabled when ddgs unavailable) -->
	<button
		on:click={toggleSearch}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{!ddgsAvailable ? disabledStyle : ($webSearchEnabled ? activeStyle : inactiveStyle)}"
		title={searchTooltip}
		aria-label="Toggle web search"
		aria-pressed={$webSearchEnabled}
		disabled={!ddgsAvailable}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<circle cx="11" cy="11" r="8" />
			<path d="M21 21l-4.35-4.35" />
		</svg>
		<span class="hidden sm:inline">Search</span>
	</button>

	<!-- S68: Toggle Cache -->
	<button
		on:click={toggleCache}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{$cacheEnabled ? activeStyle : inactiveStyle}"
		title="Toggle semantic cache (exact + embedding match)"
		aria-label="Toggle semantic cache"
		aria-pressed={$cacheEnabled}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<ellipse cx="12" cy="5" rx="9" ry="3" />
			<path d="M21 12c0 1.66-4.03 3-9 3s-9-1.34-9-3" />
			<path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
		</svg>
		<span class="hidden sm:inline">Cache</span>
	</button>

	<!-- S69: Toggle Cascading -->
	<button
		on:click={toggleCascading}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{$cascadingEnabled ? activeStyle : inactiveStyle}"
		title="Toggle cascading inference (multi-tier model routing)"
		aria-label="Toggle cascading inference"
		aria-pressed={$cascadingEnabled}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<path d="M12 2L2 7l10 5 10-5-10-5z" />
			<path d="M2 17l10 5 10-5" />
			<path d="M2 12l10 5 10-5" />
		</svg>
		<span class="hidden sm:inline">Cascade</span>
	</button>

	<!-- S84: Toggle Prompt Enhancement (Onion button) -->
	<button
		on:click={togglePromptEnhance}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{$promptEnhanceEnabled ? activeStyle : inactiveStyle}"
		title="Toggle prompt enhancement (optimize prompts before sending)"
		aria-label="Toggle prompt enhancement"
		aria-pressed={$promptEnhanceEnabled}
	>
		<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
			stroke-linecap="round" stroke-linejoin="round">
			<ellipse cx="12" cy="14" rx="8" ry="7" />
			<ellipse cx="12" cy="14" rx="5.5" ry="5" />
			<ellipse cx="12" cy="14" rx="3" ry="3" />
			<path d="M12 7V3" />
			<path d="M10 4.5c1-1 3-1 4 0" />
		</svg>
		<span class="hidden sm:inline">Opti</span>
	</button>

	<!-- S86: Toggle Humanize -->
	<button
		on:click={toggleHumanize}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{$humanizeEnabled ? activeStyle : inactiveStyle}"
		title="Toggle humanizer post-processing (make output more natural)"
		aria-label="Toggle humanizer"
		aria-pressed={$humanizeEnabled}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
			stroke-linecap="round" stroke-linejoin="round">
			<path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
		</svg>
		<span class="hidden sm:inline">Human</span>
	</button>

	<!-- S117: Toggle Quick Sandbox -->
	<button
		on:click={toggleQuickSandbox}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{!qsAvailable ? disabledStyle : ($quickSandboxEnabled ? activeStyle : inactiveStyle)}"
		title={qsAvailable ? 'Toggle sandboxed code execution (isolate LLM tool calls)' : 'Sandbox not available (install bubblewrap)'}
		aria-label="Toggle quick sandbox"
		aria-pressed={$quickSandboxEnabled}
		disabled={!qsAvailable}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
			stroke-linecap="round" stroke-linejoin="round">
			<rect x="3" y="3" width="18" height="18" rx="2" />
			<path d="M9 3v18" />
			<path d="M15 3v18" />
			<path d="M3 9h18" />
			<path d="M3 15h18" />
		</svg>
		<span class="hidden sm:inline">Sandbox</span>
	</button>

	<!-- S118: Toggle Chat Coding Agent (sage accent to distinguish) -->
	<button
		on:click={toggleChatCoding}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{!ccAvailable ? disabledStyle : ($chatCodingEnabled
			? 'background-color: var(--oo-sage-bg); color: var(--oo-sage); border: 1px solid var(--oo-sage-bg);'
			: inactiveStyle)}"
		title={ccAvailable ? 'Toggle coding agent (multi-turn plan/implement/test/fix in sandbox)' : 'Code Agent not available (install bubblewrap)'}
		aria-label="Toggle chat coding agent"
		aria-pressed={$chatCodingEnabled}
		disabled={!ccAvailable}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
			stroke-linecap="round" stroke-linejoin="round">
			<path d="M16 18l2-2-2-2" />
			<path d="M8 6L6 8l2 2" />
			<path d="M14.5 4l-5 16" />
		</svg>
		<span class="hidden sm:inline">Code</span>
	</button>

	<!-- S131: Wipe Conversation (visible only when available + in a conversation) -->
	{#if wipeAvailable && $activeConversationId}
		<div class="w-px h-4 hidden sm:block" style="background-color: var(--oo-bd-default);" />
		<button
			on:click={handleWipeConversation}
			class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
				transition-all select-none"
			style="background-color: var(--oo-bg-surface); color: var(--oo-fg-muted); border: 1px solid var(--oo-bg-surface);
				{wipeBusy ? 'opacity: 0.5; cursor: not-allowed;' : ''}"
			title="Wipe conversation data from RAM (best-effort)"
			aria-label="Wipe conversation from RAM"
			disabled={wipeBusy}
		>
			<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
				stroke-linecap="round" stroke-linejoin="round">
				<path d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
			</svg>
			<span class="hidden sm:inline">{wipeBusy ? 'Wiping...' : 'Wipe'}</span>
		</button>
	{/if}
</div>
