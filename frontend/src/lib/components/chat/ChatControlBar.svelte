<!--
  ChatControlBar.svelte
  Horizontal control bar above the input area.
  Provides model/preset selectors and feature toggles with responsive labels.
  Labels collapse to icon-only on screens < 640px (sm breakpoint).
  All active toggles use a borderless tobacco tint (v4e palette).
  Responsive labels, unified style, ddgs availability check,
       model family grouping with parameter badges.
  Mobile responsive -- horizontal scroll overflow, touch-friendly min-height.
  Think and Search are choices for the next messages. Cache, Cascade, Human,
  Sandbox and Code are server-wide switches: each shows the state the server
  confirmed, is dimmed and announces no state while that is unknown, and
  names what the server refused, under the bar, where a state that could not
  be read can be read again. Wipe asks first, naming the conversation it
  asked about, in a dialog that shows why it failed if it does.
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
		quickSandboxEnabled,
		chatCodingEnabled,
		loadOptions,
	} from '$lib/stores/chatOptions';
	import { activeConversation, activeConversationId } from '$lib/stores/conversations';
	import { workspaceBinding } from '$lib/stores/workspaceBinding';
	import { getConversationBinding, getQuickSandboxStatus, setQuickSandbox } from '$lib/api/sandbox';
	import { getSemCacheStatus, toggleSemCache } from '$lib/api/semanticCache';
	import { getCascadingStatus, updateCascadingConfig } from '$lib/api/cascading';
	import { getHumanizerConfig, updateHumanizerConfig } from '$lib/api/humanizer';
	import { getChatCodingStatus, setChatCoding } from '$lib/api/codingAgent';
	import { getSearchConfig } from '$lib/api/search';
	import { getHardeningStatus, wipeConversation } from '$lib/api/hardening';
	import { parseApiError } from '$lib/api/errorHandler';
	import { createServerSwitch, pressed } from '$lib/switches/serverSwitch';
	import ConfirmDialog from '$lib/ds/ConfirmDialog.svelte';
	import InlineError from '$lib/ds/InlineError.svelte';

	let loaded = false;

	// Track whether duckduckgo-search is installed
	let ddgsAvailable = true;

	// Whether the semantic cache, the quick sandbox and the coding agent can
	// run on the server, read with their state.
	let cacheAvailable = true;
	let qsAvailable = false;
	let ccAvailable = false;

	// The server-wide switches. Each shows the state the server confirmed:
	// unknown until it is read, unchanged until the server answers a change,
	// and its error when the server refused or could not be reached. They
	// reach the server through the API layer, which carries the CSRF header.
	const cache = createServerSwitch({
		label: 'Semantic cache',
		read: async () => {
			const status = await getSemCacheStatus();
			cacheAvailable = status.available;
			return status.enabled;
		},
		write: async () => (await toggleSemCache()).enabled,
	});
	const cascade = createServerSwitch({
		label: 'Cascading',
		read: async () => (await getCascadingStatus()).enabled,
		write: async (next) => (await updateCascadingConfig({ enabled: next })).enabled,
	});
	const humanizer = createServerSwitch({
		label: 'Output humanizer',
		read: async () => (await getHumanizerConfig()).enabled,
		write: async (next) => (await updateHumanizerConfig({ enabled: next })).enabled,
	});
	const sandbox = createServerSwitch({
		label: 'Quick sandbox',
		read: async () => {
			const status = await getQuickSandboxStatus();
			qsAvailable = status.available;
			return status.enabled;
		},
		write: async (next) => (await setQuickSandbox(next)).enabled,
		adopt: (value) => quickSandboxEnabled.set(value),
		forget: () => quickSandboxEnabled.set(false),
	});
	const coding = createServerSwitch({
		label: 'Coding agent',
		read: async () => {
			const status = await getChatCodingStatus();
			ccAvailable = status.available;
			return status.enabled;
		},
		write: async (next) => (await setChatCoding(next)).enabled,
		adopt: (value) => chatCodingEnabled.set(value),
		forget: () => chatCodingEnabled.set(false),
	});

	// Conversation wipe: asked first, about the conversation open when it was
	// asked, and its failure shown.
	let wipeAvailable = false;
	let wipeOpen = false;
	let wipeBusy = false;
	let wipeError: string | null = null;
	let wipeTarget: { id: string; title: string } | null = null;

	// What turning one of the two exclusive defaults on did to the other,
	// when the second change did not follow.
	let exclusionNote: string | null = null;

	// Group models by family for the dropdown
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

	// Unified active toggle style (borderless, tobacco tint, v4e palette)
	const activeStyle = 'background-color: var(--oo-tobacco-bg); color: var(--oo-tobacco); border: 1px solid var(--oo-tobacco-bg);';
	const inactiveStyle = 'background-color: var(--oo-bg-surface); color: var(--oo-fg-muted); border: 1px solid var(--oo-bg-surface);';
	const disabledStyle = 'background-color: var(--oo-bg-surface); color: var(--oo-fg-muted); border: 1px solid var(--oo-bg-surface); opacity: 0.5; cursor: not-allowed;';

	async function readSearchAvailability() {
		try {
			ddgsAvailable = (await getSearchConfig()).ddgs_available ?? true;
		} catch {
			// If the search configuration cannot be read, assume search is there.
		}
	}

	async function readWipeAvailability() {
		try {
			wipeAvailable = (await getHardeningStatus()).conversation_wipe?.available ?? false;
		} catch {
			wipeAvailable = false;
		}
	}

	onMount(async () => {
		if ($availableModels.length === 0) {
			await loadOptions();
		}
		loaded = true;
		await Promise.all([
			cache.load(),
			cascade.load(),
			humanizer.load(),
			sandbox.load(),
			coding.load(),
			readSearchAvailability(),
			readWipeAvailability(),
		]);
	});

	function toggleThinking() {
		thinkingEnabled.update((v) => !v);
	}

	function toggleSearch() {
		if (!ddgsAvailable) return;
		webSearchEnabled.update((v) => !v);
	}

	// The quick sandbox and the coding agent exclude each other: turning one
	// on turns the other off first, on the server, and stops there if the
	// server keeps it on. When the second change does not follow, the first
	// is named, since it changed a server-wide default too.
	async function toggleQuickSandbox() {
		if (!qsAvailable) return;
		exclusionNote = null;
		let turnedOff = false;
		if ($sandbox.value === false && $coding.value === true) {
			await coding.toggle();
			if (coding.current().value !== false) return;
			turnedOff = true;
		}
		await sandbox.toggle();
		if (turnedOff && sandbox.current().value !== true) {
			exclusionNote = 'The coding agent default was turned off first; the quick sandbox default did not turn on.';
		}
	}

	async function toggleChatCoding() {
		if (!ccAvailable) return;
		exclusionNote = null;
		let turnedOff = false;
		if ($coding.value === false && $sandbox.value === true) {
			await sandbox.toggle();
			if (sandbox.current().value !== false) return;
			turnedOff = true;
		}
		await coding.toggle();
		if (turnedOff && coding.current().value !== true) {
			exclusionNote = 'The quick sandbox default was turned off first; the coding agent default did not turn on.';
		}
	}

	// The style of a server switch's pill: dimmed while its state is unknown.
	function switchStyle(value: boolean | null, available = true): string {
		if (!available || value === null) return disabledStyle;
		return value ? activeStyle : inactiveStyle;
	}

	// A server switch's tooltip: what it is, and that its state is not known
	// while it is not.
	function switchTitle(what: string, value: boolean | null): string {
		return value === null ? `${what} (state not known yet)` : what;
	}

	// Compute search toggle tooltip
	$: searchTooltip = ddgsAvailable
		? 'Toggle web search (DuckDuckGo)'
		: 'Install duckduckgo-search to enable (pip install duckduckgo-search)';

	// The workspace bound to the active conversation feeds the chip in the
	// bar. The store is conversation-scoped: switching clears it at once
	// and late responses from a left conversation are dropped, so the
	// indication never bleeds across conversations.
	$: void workspaceBinding.refreshFor(
		$activeConversationId ? String($activeConversationId) : null,
		getConversationBinding
	);

	// The wipe is asked about the conversation open now: its id and title are
	// kept, and the confirmation wipes that one, wherever the page is by then.
	function askWipe() {
		if (!$activeConversationId) return;
		wipeTarget = {
			id: String($activeConversationId),
			title: $activeConversation?.title?.trim() || 'Untitled conversation',
		};
		wipeError = null;
		wipeOpen = true;
	}

	// Closing leaves a running wipe running; its failure is then shown under
	// the bar.
	function closeWipe() {
		wipeOpen = false;
		if (!wipeBusy) wipeError = null;
	}

	// Another conversation opened while the question was asked: the question
	// no longer names what is on the page, so it is withdrawn.
	$: if (wipeOpen && !wipeBusy && wipeTarget && String($activeConversationId ?? '') !== wipeTarget.id) {
		closeWipe();
	}

	async function runWipe() {
		const target = wipeTarget;
		if (!target || wipeBusy) return;
		wipeBusy = true;
		wipeError = null;
		try {
			await wipeConversation(target.id);
			wipeOpen = false;
		} catch (e) {
			wipeError = parseApiError(e, 'wiping the conversation').message;
		} finally {
			wipeBusy = false;
		}
	}
</script>

<!-- Horizontal scroll on mobile, no-wrap to prevent overflow stacking -->
<div class="flex items-center gap-2 px-1 py-1.5 overflow-x-auto touch-scroll-x mobile-hide-scrollbar"
	style="min-height: 36px; -ms-overflow-style: none; scrollbar-width: none;">
	<!-- Model selector with family grouping and param badges -->
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

	<!-- Semantic cache (server-wide) -->
	<button
		on:click={() => cache.toggle()}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style={switchStyle($cache.value, cacheAvailable)}
		title={cacheAvailable
			? switchTitle('Semantic cache (exact + embedding match): a server-wide setting, until the server restarts', $cache.value)
			: 'Semantic cache not available on the server'}
		aria-label="Toggle semantic cache"
		aria-pressed={pressed($cache.value)}
		aria-busy={$cache.pending}
		disabled={!cacheAvailable || $cache.value === null || $cache.pending}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<ellipse cx="12" cy="5" rx="9" ry="3" />
			<path d="M21 12c0 1.66-4.03 3-9 3s-9-1.34-9-3" />
			<path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
		</svg>
		<span class="hidden sm:inline">Cache</span>
	</button>

	<!-- Cascading (server-wide, saved) -->
	<button
		on:click={() => cascade.toggle()}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style={switchStyle($cascade.value)}
		title={switchTitle('Cascading inference (multi-tier model routing): a server-wide setting, saved', $cascade.value)}
		aria-label="Toggle cascading inference"
		aria-pressed={pressed($cascade.value)}
		aria-busy={$cascade.pending}
		disabled={$cascade.value === null || $cascade.pending}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<path d="M12 2L2 7l10 5 10-5-10-5z" />
			<path d="M2 17l10 5 10-5" />
			<path d="M2 12l10 5 10-5" />
		</svg>
		<span class="hidden sm:inline">Cascade</span>
	</button>

	<!-- Output humanizer (server-wide) -->
	<button
		on:click={() => humanizer.toggle()}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style={switchStyle($humanizer.value)}
		title={switchTitle('Humanizer post-processing (more natural output): a server-wide setting, until the server restarts', $humanizer.value)}
		aria-label="Toggle humanizer"
		aria-pressed={pressed($humanizer.value)}
		aria-busy={$humanizer.pending}
		disabled={$humanizer.value === null || $humanizer.pending}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
			stroke-linecap="round" stroke-linejoin="round">
			<path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
		</svg>
		<span class="hidden sm:inline">Human</span>
	</button>

	<!-- Quick sandbox (server-wide default) -->
	<button
		on:click={toggleQuickSandbox}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style={switchStyle($sandbox.value, qsAvailable)}
		title={qsAvailable
			? switchTitle('Sandboxed code execution (isolate LLM tool calls): the server-wide default, until the server restarts', $sandbox.value)
			: 'Sandbox not available (install bubblewrap)'}
		aria-label="Toggle quick sandbox"
		aria-pressed={pressed($sandbox.value)}
		aria-busy={$sandbox.pending || $coding.pending}
		disabled={!qsAvailable || $sandbox.value === null || $sandbox.pending || $coding.pending}
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

	<!-- Coding agent (server-wide default; sage accent to distinguish) -->
	<button
		on:click={toggleChatCoding}
		class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
			transition-all select-none"
		style="{$coding.value === true && ccAvailable
			? 'background-color: var(--oo-sage-bg); color: var(--oo-sage); border: 1px solid var(--oo-sage-bg);'
			: switchStyle($coding.value, ccAvailable)}"
		title={ccAvailable
			? switchTitle('Coding agent (multi-turn plan/implement/test/fix in sandbox): the server-wide default, until the server restarts', $coding.value)
			: 'Code Agent not available (install bubblewrap)'}
		aria-label="Toggle chat coding agent"
		aria-pressed={pressed($coding.value)}
		aria-busy={$coding.pending || $sandbox.pending}
		disabled={!ccAvailable || $coding.value === null || $coding.pending || $sandbox.pending}
	>
		<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
			stroke-linecap="round" stroke-linejoin="round">
			<path d="M16 18l2-2-2-2" />
			<path d="M8 6L6 8l2 2" />
			<path d="M14.5 4l-5 16" />
		</svg>
		<span class="hidden sm:inline">Code</span>
	</button>

	<!-- Workspace bound to this conversation (indicator, not a toggle) -->
	{#if $workspaceBinding.sessionId}
		<span
			class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0 select-none"
			style="background-color: var(--oo-bg-surface); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);"
			title="Workspace bound to this conversation: {$workspaceBinding.sessionId}"
			aria-label="Workspace bound to this conversation: {$workspaceBinding.sessionId}"
			role="status"
		>
			<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
				stroke-linecap="round" stroke-linejoin="round">
				<path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71" />
				<path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71" />
			</svg>
			<span>{$workspaceBinding.sessionId.slice(0, 8)}</span>
		</span>
	{/if}

	<!-- Wipe Conversation (visible only when available + in a conversation) -->
	{#if wipeAvailable && $activeConversationId}
		<div class="w-px h-4 hidden sm:block" style="background-color: var(--oo-bd-default);" />
		<button
			on:click={askWipe}
			class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs shrink-0
				transition-all select-none"
			style="background-color: var(--oo-bg-surface); color: var(--oo-fg-muted); border: 1px solid var(--oo-bg-surface);
				{wipeBusy ? 'opacity: 0.5; cursor: not-allowed;' : ''}"
			title="Wipe conversation data from RAM (best-effort)"
			aria-label="Wipe conversation from RAM"
			aria-haspopup="dialog"
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

<!-- What the server refused or could not say, under the bar; a state that
     could not be read can be read again from here. -->
{#if $cache.error || $cascade.error || $humanizer.error || $sandbox.error || $coding.error || exclusionNote || (wipeError && !wipeOpen)}
	<div class="flex flex-col gap-1 px-1 pb-1">
		<InlineError message={$cache.error} onRetry={$cache.value === null ? cache.load : undefined} retrying={$cache.pending} />
		<InlineError message={$cascade.error} onRetry={$cascade.value === null ? cascade.load : undefined} retrying={$cascade.pending} />
		<InlineError message={$humanizer.error} onRetry={$humanizer.value === null ? humanizer.load : undefined} retrying={$humanizer.pending} />
		<InlineError message={$sandbox.error} onRetry={$sandbox.value === null ? sandbox.load : undefined} retrying={$sandbox.pending} />
		<InlineError message={$coding.error} onRetry={$coding.value === null ? coding.load : undefined} retrying={$coding.pending} />
		<InlineError message={exclusionNote} />
		{#if !wipeOpen}
			<InlineError message={wipeError} />
		{/if}
	</div>
{/if}

<ConfirmDialog
	open={wipeOpen}
	title="Wipe this conversation from memory?"
	message={wipeTarget
		? `"${wipeTarget.title}": its messages are zeroed in the server's memory, best-effort. This cannot be undone.`
		: ''}
	confirmLabel="Wipe"
	cancelLabel="Keep it"
	danger
	busy={wipeBusy}
	error={wipeError}
	onConfirm={runWipe}
	onCancel={closeWipe}
/>
