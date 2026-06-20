<!--
  DashboardHome.svelte (S107)
  Landing screen when no conversation is selected.
  Shows: system health, available models, recent conversations, quick actions.
  Uses existing API endpoints: /api/health, /api/models, /api/conversations.
  All styles use --oo-* CSS variables exclusively.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { createNewConversation } from '$lib/stores/conversations';
	import { backendStatus, backendVersion, backendModules } from '$lib/stores/health';
	import { toastError } from '$lib/stores/notifications';
	import type { ModelInfo, ConversationSummary } from '$lib/types';

	let models: ModelInfo[] = [];
	let recentChats: ConversationSummary[] = [];
	let loading = true;

	const API_BASE = import.meta.env.VITE_API_URL ?? '';

	async function fetchDashboardData() {
		loading = true;
		try {
			const [modelsResp, chatsResp] = await Promise.allSettled([
				fetch(`${API_BASE}/api/models`).then((r) => r.ok ? r.json() : null),
				fetch(`${API_BASE}/api/conversations?limit=5`).then((r) => r.ok ? r.json() : null),
			]);

			if (modelsResp.status === 'fulfilled' && modelsResp.value) {
				models = modelsResp.value.models ?? [];
			}
			if (chatsResp.status === 'fulfilled' && chatsResp.value) {
				// API returns array directly or { conversations: [...] }
				recentChats = Array.isArray(chatsResp.value)
					? chatsResp.value.slice(0, 5)
					: (chatsResp.value.conversations ?? []).slice(0, 5);
			}
		} catch {
			// Silently handle — dashboard is informational
		} finally {
			loading = false;
		}
	}

	async function handleNewChat() {
		try {
			const id = await createNewConversation();
			goto(`/chat/${id}`);
		} catch {
			toastError('Failed to create conversation');
		}
	}

	function handleOpenChat(id: string) {
		goto(`/chat/${id}`);
	}

	function formatDate(dateStr: string | null): string {
		if (!dateStr) return '';
		const d = new Date(dateStr);
		const now = new Date();
		const diff = now.getTime() - d.getTime();
		if (diff < 60_000) return 'just now';
		if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
		if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
		return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
	}

	$: moduleTotal = Object.keys($backendModules).length;
	$: moduleActive = Object.values($backendModules).filter(Boolean).length;

	onMount(fetchDashboardData);
</script>

<div class="dash-root">
	<div class="dash-container">

		<!-- Logo + title -->
		<div class="dash-header">
			<img src="/bousier-oignon.png" alt="Opti-Oignon" class="dash-logo oo-logo-adaptive" />
			<div>
				<h1 class="dash-title">Opti-Oignon</h1>
				<p class="dash-subtitle">Local LLM orchestration platform</p>
			</div>
		</div>

		<!-- Quick actions -->
		<div class="dash-actions">
			<button class="dash-action-btn dash-action-primary" on:click={handleNewChat}>
				<svg class="dash-action-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M12 4v16m8-8H4" />
				</svg>
				New conversation
			</button>
			<a href="/benchmark" class="dash-action-btn">
				<svg class="dash-action-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
					<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
				</svg>
				Run benchmark
			</a>
			<a href="/settings" class="dash-action-btn">
				<svg class="dash-action-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
					<path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
				</svg>
				Settings
			</a>
		</div>

		<!-- Cards grid -->
		<div class="dash-grid">

			<!-- System health card -->
			<div class="dash-card">
				<div class="dash-card-header">
					<svg class="dash-card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
					</svg>
					<span class="dash-card-title">System Health</span>
				</div>
				<div class="dash-card-body">
					<div class="dash-stat-row">
						<span class="dash-stat-label">Backend</span>
						<span class="dash-stat-value">
							<span class="dash-dot" style="background-color: {$backendStatus === 'connected' ? 'var(--oo-success)' : $backendStatus === 'degraded' ? 'var(--oo-warning)' : 'var(--oo-error)'};" />
							{$backendStatus === 'connected' ? 'Connected' : $backendStatus === 'degraded' ? 'Degraded' : 'Disconnected'}
						</span>
					</div>
					{#if $backendVersion}
						<div class="dash-stat-row">
							<span class="dash-stat-label">Version</span>
							<span class="dash-stat-value dash-mono">v{$backendVersion}</span>
						</div>
					{/if}
					{#if moduleTotal > 0}
						<div class="dash-stat-row">
							<span class="dash-stat-label">Modules</span>
							<span class="dash-stat-value">{moduleActive}/{moduleTotal} active</span>
						</div>
					{/if}
				</div>
			</div>

			<!-- Models card -->
			<div class="dash-card">
				<div class="dash-card-header">
					<svg class="dash-card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
					</svg>
					<span class="dash-card-title">Models</span>
					{#if models.length > 0}
						<span class="dash-card-badge">{models.length}</span>
					{/if}
				</div>
				<div class="dash-card-body">
					{#if loading}
						<span class="dash-loading">Loading...</span>
					{:else if models.length === 0}
						<span class="dash-empty-hint">No models found. Is Ollama running?</span>
					{:else}
						<div class="dash-model-list">
							{#each models.slice(0, 6) as model}
								<div class="dash-model-item">
									<span class="dash-model-name">{model.name}</span>
									{#if model.parameter_size}
										<span class="dash-model-size">{model.parameter_size}</span>
									{/if}
								</div>
							{/each}
							{#if models.length > 6}
								<span class="dash-more">+{models.length - 6} more</span>
							{/if}
						</div>
					{/if}
				</div>
			</div>

			<!-- Recent chats card -->
			<div class="dash-card dash-card-wide">
				<div class="dash-card-header">
					<svg class="dash-card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
					</svg>
					<span class="dash-card-title">Recent Conversations</span>
				</div>
				<div class="dash-card-body">
					{#if loading}
						<span class="dash-loading">Loading...</span>
					{:else if recentChats.length === 0}
						<span class="dash-empty-hint">No conversations yet. Start a new one!</span>
					{:else}
						<div class="dash-chat-list">
							{#each recentChats as chat}
								<button class="dash-chat-item" on:click={() => handleOpenChat(chat.id)}>
									<span class="dash-chat-title">{chat.title || 'Untitled'}</span>
									<div class="dash-chat-meta">
										{#if chat.message_count > 0}
											<span>{chat.message_count} msg{chat.message_count > 1 ? 's' : ''}</span>
										{/if}
										{#if chat.updated_at}
											<span>{formatDate(chat.updated_at)}</span>
										{/if}
									</div>
								</button>
							{/each}
						</div>
					{/if}
				</div>
			</div>

		</div>

		<!-- Keyboard shortcut hint -->
		<div class="dash-hint">
			Press <kbd class="dash-kbd">?</kbd> for keyboard shortcuts
		</div>

	</div>
</div>

<style>
	.dash-root {
		height: 100%;
		overflow-y: auto;
		display: flex;
		align-items: flex-start;
		justify-content: center;
		padding: 2rem 1rem;
	}

	.dash-container {
		width: 100%;
		max-width: 640px;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	/* Header */
	.dash-header {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.dash-logo {
		width: 88px;
		height: 88px;
		object-fit: contain;
		border-radius: 10px;
	}

	.dash-title {
		font-size: 1.375rem;
		font-weight: 700;
		color: var(--oo-fg-primary);
		margin: 0;
		line-height: 1.3;
	}

	.dash-subtitle {
		font-size: 0.8125rem;
		color: var(--oo-fg-muted);
		margin: 0;
	}

	/* Quick actions */
	.dash-actions {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}

	.dash-action-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.875rem;
		border-radius: 8px;
		border: 1px solid var(--oo-bd-subtle);
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-secondary);
		font-size: 0.8125rem;
		font-weight: 500;
		text-decoration: none;
		cursor: pointer;
		transition: background-color 0.15s ease, border-color 0.15s ease;
	}

	.dash-action-btn:hover {
		background-color: var(--oo-bg-overlay);
		border-color: var(--oo-bd-default);
	}

	.dash-action-primary {
		background-color: var(--oo-btn-primary-bg);
		color: var(--oo-btn-primary-fg);
		border-color: transparent;
	}

	.dash-action-primary:hover {
		opacity: 0.9;
		border-color: transparent;
	}

	.dash-action-icon {
		width: 16px;
		height: 16px;
		flex-shrink: 0;
	}

	/* Grid */
	.dash-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
	}

	@media (max-width: 480px) {
		.dash-grid {
			grid-template-columns: 1fr;
		}
		.dash-card-wide {
			grid-column: 1;
		}
	}

	.dash-card-wide {
		grid-column: 1 / -1;
	}

	/* Card */
	.dash-card {
		border-radius: 10px;
		border: 1px solid var(--oo-bd-subtle);
		background-color: var(--oo-bg-elevated);
		overflow: hidden;
	}

	.dash-card-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem 1rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
	}

	.dash-card-icon {
		width: 16px;
		height: 16px;
		color: var(--oo-fg-muted);
		flex-shrink: 0;
	}

	.dash-card-title {
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.dash-card-badge {
		margin-left: auto;
		padding: 0 0.375rem;
		border-radius: 6px;
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-muted);
		font-size: 0.6875rem;
		font-weight: 600;
	}

	.dash-card-body {
		padding: 0.75rem 1rem;
	}

	/* Stats */
	.dash-stat-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.25rem 0;
	}

	.dash-stat-label {
		font-size: 0.8125rem;
		color: var(--oo-fg-muted);
	}

	.dash-stat-value {
		font-size: 0.8125rem;
		color: var(--oo-fg-primary);
		display: inline-flex;
		align-items: center;
		gap: 0.375rem;
	}

	.dash-mono {
		font-family: var(--oo-font-mono);
	}

	.dash-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	/* Models */
	.dash-model-list {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}

	.dash-model-item {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.1875rem 0;
	}

	.dash-model-name {
		font-size: 0.8125rem;
		color: var(--oo-fg-secondary);
		font-family: var(--oo-font-mono);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.dash-model-size {
		font-size: 0.6875rem;
		color: var(--oo-fg-faint);
		flex-shrink: 0;
		margin-left: 0.5rem;
	}

	.dash-more {
		font-size: 0.75rem;
		color: var(--oo-fg-faint);
		padding-top: 0.25rem;
	}

	/* Chats */
	.dash-chat-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.dash-chat-item {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.625rem;
		border-radius: 6px;
		border: none;
		background: transparent;
		cursor: pointer;
		text-align: left;
		width: 100%;
		transition: background-color 0.1s ease;
	}

	.dash-chat-item:hover {
		background-color: var(--oo-bg-overlay);
	}

	.dash-chat-title {
		font-size: 0.8125rem;
		color: var(--oo-fg-secondary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1;
		min-width: 0;
	}

	.dash-chat-meta {
		display: flex;
		gap: 0.5rem;
		flex-shrink: 0;
		margin-left: 0.75rem;
		font-size: 0.6875rem;
		color: var(--oo-fg-faint);
	}

	/* Misc */
	.dash-loading {
		font-size: 0.8125rem;
		color: var(--oo-fg-faint);
	}

	.dash-empty-hint {
		font-size: 0.8125rem;
		color: var(--oo-fg-faint);
	}

	.dash-hint {
		text-align: center;
		font-size: 0.75rem;
		color: var(--oo-fg-faint);
		padding-top: 0.5rem;
	}

	.dash-kbd {
		display: inline;
		padding: 0.125rem 0.375rem;
		border-radius: 4px;
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-bd-subtle);
		font-size: 0.75rem;
		color: var(--oo-fg-muted);
		font-family: var(--oo-font-mono);
	}
</style>
