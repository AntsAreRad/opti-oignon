<!--
  StatusFooter.svelte
  Optional thin status bar at the bottom of the shell (spec 8.5). Shows
  the active mode, resolved model, context window and token usage for the
  current chat. Toggleable via Appearance (statusFooterVisible) and
  auto-hidden off /chat or when no conversation is active. Resolves N1
  (system status visibility). Latency is surfaced once a metric source is
  wired (LiveMetrics work).
-->
<script lang="ts">
	import { page } from '$app/stores';
	import { statusFooterVisible } from '$lib/stores/preferences';
	import { currentMode } from '$lib/stores/securityMode';
	import { selectedModel } from '$lib/stores/chatOptions';
	import { activeConversation, activeConversationId } from '$lib/stores/conversations';
	import { isStreaming } from '$lib/stores/chat';
	import { getContextHealth } from '$lib/api/context';
	import type { ContextHealthResponse } from '$lib/types';

	let data: ContextHealthResponse | null = null;

	$: onChat = ($page.url?.pathname ?? '').startsWith('/chat');
	$: visible = $statusFooterVisible && onChat && !!$activeConversationId;

	// Refresh when the conversation, model, or streaming state changes.
	$: if (visible && ($activeConversationId || $selectedModel || !$isStreaming)) {
		loadContext();
	}

	async function loadContext() {
		if (!$activeConversationId) {
			data = null;
			return;
		}
		try {
			data = await getContextHealth(
				$activeConversationId || undefined,
				$selectedModel || undefined
			);
		} catch {
			data = null;
		}
	}

	$: modeLabel = $currentMode === 'bulbe' ? 'Bulbe' : 'Daily';
	$: tokens = data?.current_conversation?.estimated_tokens ?? 0;
	$: budget = data?.current_conversation?.model_context_window ?? 0;
	$: model =
		data?.current_conversation?.model ||
		$selectedModel ||
		$activeConversation?.model ||
		'default';

	function fmt(n: number): string {
		if (n < 1000) return String(n);
		if (n < 10000) return `${(n / 1000).toFixed(1)}k`;
		return `${(n / 1000).toFixed(0)}k`;
	}
	function shortModel(name: string): string {
		const base = name.split(':')[0];
		return base.length > 18 ? base.slice(0, 16) + '...' : name;
	}
</script>

{#if visible}
	<footer class="oo-status-footer" aria-label="Session status">
		<span class="oo-status-item">
			<span class="oo-status-key">Mode</span>
			<span>{modeLabel}</span>
		</span>
		<span class="oo-status-sep" aria-hidden="true"></span>
		<span class="oo-status-item" title={model}>
			<span class="oo-status-key">Model</span>
			<span>{shortModel(model)}</span>
		</span>
		<span class="oo-status-sep" aria-hidden="true"></span>
		<span class="oo-status-item" title="{tokens} / {budget} tokens">
			<span class="oo-status-key">Ctx</span>
			<span class="tabular-nums">{fmt(tokens)}/{fmt(budget)}</span>
		</span>
		<span class="oo-status-sep" aria-hidden="true"></span>
		<span class="oo-status-item">
			<span class="oo-status-key">Tokens</span>
			<span class="tabular-nums">{tokens} used</span>
		</span>
	</footer>
{/if}

<style>
	.oo-status-footer {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		padding: var(--oo-space-1) var(--oo-space-4);
		border-top: 1px solid var(--oo-bd-subtle);
		background-color: var(--oo-bg-surface);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-2xs);
		line-height: 1;
		white-space: nowrap;
		overflow-x: auto;
		flex-shrink: 0;
	}
	.oo-status-item {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
	.oo-status-key {
		color: var(--oo-fg-tertiary);
		text-transform: uppercase;
		letter-spacing: var(--oo-tracking-wide);
	}
	.oo-status-sep {
		width: 1px;
		height: 12px;
		background-color: var(--oo-bd-subtle);
		flex-shrink: 0;
	}
</style>
