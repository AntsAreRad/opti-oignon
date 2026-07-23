<!--
  ProxySettingsPanel.svelte -- Anonymous Web Search / Tor Proxy settings.

  Sections:
  1. Proxy mode selector (Off / Tor / Custom SOCKS5)
  2. Proxy status indicator (reachable, latency, exit IP)
  3. PII sanitization toggle + preview
  4. Retry and timeout configuration
  5. Search stats overview
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getProxyStatus,
		getProxyConfig,
		updateProxyConfig,
		previewPII,
		getSearchConfig,
	} from '$lib/api/search';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		ProxyStatusResponse,
		ProxyConfigResponse,
		PIISanitizePreviewResponse,
		SearchConfigResponse,
	} from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	// Config
	let localMode = 'off';
	let localProxyUrl = '';
	let localProxyTimeout = 15;
	let localMaxRetries = 3;
	let localPiiEnabled = true;
	let saving = false;

	// Proxy status
	let proxyStatus: ProxyStatusResponse | null = null;
	let checkingProxy = false;

	// PII preview
	let piiTestQuery = 'error at user@example.com on 192.168.1.1 in /home/leon/project';
	let piiPreview: PIISanitizePreviewResponse | null = null;
	let previewingPII = false;

	// Stats
	let searchConfig: SearchConfigResponse | null = null;

	// Track ddgs availability
	$: ddgsAvailable = searchConfig?.ddgs_available ?? true;

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadData);

	async function loadData() {
		loading = true;
		error = '';
		try {
			const [config, stats] = await Promise.all([
				getProxyConfig(),
				getSearchConfig(),
			]);
			applyConfig(config);
			searchConfig = stats;
		} catch (e: any) {
			error = e?.detail || e?.message || 'Failed to load search settings';
		} finally {
			loading = false;
		}
	}

	function applyConfig(config: ProxyConfigResponse) {
		localMode = config.mode;
		localProxyUrl = config.proxy_url || '';
		localProxyTimeout = config.proxy_timeout;
		localMaxRetries = config.max_retries;
		localPiiEnabled = config.pii_sanitization_enabled;
	}

	// -------------------------------------------------------------------------
	// Save
	// -------------------------------------------------------------------------

	async function saveConfig() {
		saving = true;
		try {
			const result = await updateProxyConfig({
				mode: localMode,
				proxy_url: localMode === 'custom' ? localProxyUrl : null,
				proxy_timeout: localProxyTimeout,
				max_retries: localMaxRetries,
				pii_sanitization_enabled: localPiiEnabled,
			});
			applyConfig(result);
			toastSuccess('Search proxy settings saved');
		} catch (e: any) {
			toastError(e?.detail || 'Failed to save settings');
		} finally {
			saving = false;
		}
	}

	// -------------------------------------------------------------------------
	// Proxy check
	// -------------------------------------------------------------------------

	async function checkProxy() {
		checkingProxy = true;
		proxyStatus = null;
		try {
			proxyStatus = await getProxyStatus();
		} catch (e: any) {
			toastError(e?.detail || 'Proxy check failed');
		} finally {
			checkingProxy = false;
		}
	}

	// -------------------------------------------------------------------------
	// PII preview
	// -------------------------------------------------------------------------

	async function runPIIPreview() {
		if (!piiTestQuery.trim()) return;
		previewingPII = true;
		piiPreview = null;
		try {
			piiPreview = await previewPII(piiTestQuery);
		} catch (e: any) {
			toastError(e?.detail || 'PII preview failed');
		} finally {
			previewingPII = false;
		}
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	function modeLabel(mode: string): string {
		switch (mode) {
			case 'off': return 'Direct (no proxy)';
			case 'tor': return 'Tor (socks5h://localhost:9050)';
			case 'custom': return 'Custom SOCKS5';
			default: return mode;
		}
	}
</script>

<!-- ===================================================================== -->
<!-- Template -->
<!-- ===================================================================== -->

<div class="panel" style="background: var(--oo-bg-surface); border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-lg); padding: 1.25rem;">
	<div style="display: flex; align-items: center; justify-content: space-between; margin: 0 0 1rem 0;">
		<h3 style="margin: 0; color: var(--oo-fg-primary); font-size: 1rem; font-weight: 600;">
			Web Search &amp; Privacy
		</h3>
		<!-- Compact proxy status badge -->
		{#if !loading && !error}
			<span style="display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.2rem 0.5rem;
				border-radius: 9999px; font-size: 0.7rem; font-weight: 500;
				{localMode === 'off'
					? 'background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-default);'
					: proxyStatus?.reachable
						? 'background-color: var(--oo-success-bg); color: var(--oo-status-success); border: 1px solid var(--oo-success-bd);'
						: 'background-color: var(--oo-error-bg); color: var(--oo-status-error); border: 1px solid var(--oo-error-bd);'
				}">
				<span style="width: 6px; height: 6px; border-radius: 50%; display: inline-block;
					background-color: {localMode === 'off'
						? 'var(--oo-fg-muted)'
						: proxyStatus?.reachable
							? 'var(--oo-status-success)'
							: 'var(--oo-status-error)'};" />
				{#if localMode === 'off'}
					No proxy
				{:else if proxyStatus?.reachable}
					Connected
				{:else if proxyStatus && !proxyStatus.reachable}
					Disconnected
				{:else}
					Not checked
				{/if}
			</span>
		{/if}
	</div>

	<!-- Warning when duckduckgo-search is not installed -->
	{#if !loading && !ddgsAvailable}
		<div style="margin-bottom: 1rem; padding: 0.6rem 0.75rem; border-radius: var(--oo-radius-md);
			background-color: var(--oo-warning-bg); border: 1px solid var(--oo-warning-bd);">
			<div style="display: flex; align-items: flex-start; gap: 0.5rem;">
				<svg style="width: 1rem; height: 1rem; color: var(--oo-status-warning); flex-shrink: 0; margin-top: 0.1rem;"
					fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M12 9v2m0 4h.01M10.29 3.86l-8.36 14.31A1 1 0 002.82 20h18.36a1 1 0 00.89-1.54L13.71 3.86a1 1 0 00-1.78 0z"
						stroke-linecap="round" stroke-linejoin="round" />
				</svg>
				<div>
					<p style="color: var(--oo-status-warning); font-size: 0.8rem; font-weight: 600; margin: 0 0 0.25rem 0;">
						Web search unavailable
					</p>
					<p style="color: var(--oo-fg-tertiary); font-size: 0.75rem; margin: 0 0 0.4rem 0;">
						The <code style="font-size: 0.7rem; padding: 0.1rem 0.3rem; border-radius: 3px;
							background-color: var(--oo-bg-tertiary);">duckduckgo-search</code> package is not installed.
						You can still configure proxy settings for when it becomes available.
					</p>
					<code style="display: block; font-size: 0.75rem; padding: 0.35rem 0.5rem; border-radius: var(--oo-radius-md);
						background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); user-select: all;">pip install duckduckgo-search</code>
				</div>
			</div>
		</div>
	{/if}

	{#if loading}
		<p style="color: var(--oo-fg-tertiary);">Loading...</p>
	{:else if error}
		<p style="color: var(--oo-status-error);">{error}</p>
		<button
			style="margin-top: 0.5rem; padding: 0.4rem 0.8rem; border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default); cursor: pointer;"
			on:click={loadData}
		>
			Retry
		</button>
	{:else}
		<!-- ============================================================= -->
		<!-- Proxy Mode -->
		<!-- ============================================================= -->
		<div style="margin-bottom: 1rem;">
			<label style="display: block; color: var(--oo-fg-secondary); font-size: 0.8rem; margin-bottom: 0.4rem; font-weight: 500;">
				Proxy Mode
			</label>
			<div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
				{#each ['off', 'tor', 'custom'] as mode}
					<button
						style="padding: 0.4rem 0.75rem; border-radius: var(--oo-radius-md); border: 1px solid {localMode === mode ? 'var(--oo-accent-primary)' : 'var(--oo-bd-default)'}; background: {localMode === mode ? 'var(--oo-accent-primary)' : 'var(--oo-bg-tertiary)'}; color: {localMode === mode ? 'var(--oo-bg-surface)' : 'var(--oo-fg-secondary)'}; cursor: pointer; font-size: 0.8rem; font-weight: 500;"
						on:click={() => localMode = mode}
					>
						{modeLabel(mode)}
					</button>
				{/each}
			</div>
		</div>

		<!-- Custom proxy URL -->
		{#if localMode === 'custom'}
			<div style="margin-bottom: 1rem;">
				<label style="display: block; color: var(--oo-fg-secondary); font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">
					SOCKS5 Proxy URL
				</label>
				<input
					type="text"
					bind:value={localProxyUrl}
					placeholder="socks5h://host:port"
					style="width: 100%; box-sizing: border-box; padding: 0.4rem 0.6rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-primary); font-size: 0.85rem;"
				/>
			</div>
		{/if}

		<!-- Timeout & Retries -->
		<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
			<div>
				<label style="display: block; color: var(--oo-fg-secondary); font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">
					Proxy Timeout (s)
				</label>
				<input
					type="number"
					bind:value={localProxyTimeout}
					min="5"
					max="60"
					style="width: 100%; box-sizing: border-box; padding: 0.4rem 0.6rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-primary); font-size: 0.85rem;"
				/>
			</div>
			<div>
				<label style="display: block; color: var(--oo-fg-secondary); font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">
					Max Retries
				</label>
				<input
					type="number"
					bind:value={localMaxRetries}
					min="0"
					max="10"
					style="width: 100%; box-sizing: border-box; padding: 0.4rem 0.6rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-primary); font-size: 0.85rem;"
				/>
			</div>
		</div>

		<!-- Save + Check buttons -->
		<div style="display: flex; gap: 0.5rem; margin-bottom: 1.25rem;">
			<button
				style="padding: 0.4rem 0.9rem; border-radius: var(--oo-radius-md); background: var(--oo-accent-primary); color: var(--oo-bg-surface); border: none; cursor: pointer; font-size: 0.8rem; font-weight: 500; opacity: {saving ? 0.6 : 1};"
				on:click={saveConfig}
				disabled={saving}
			>
				{saving ? 'Saving...' : 'Save'}
			</button>
			{#if localMode !== 'off'}
				<button
					style="padding: 0.4rem 0.9rem; border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default); cursor: pointer; font-size: 0.8rem; font-weight: 500; opacity: {checkingProxy ? 0.6 : 1};"
					on:click={checkProxy}
					disabled={checkingProxy}
				>
					{checkingProxy ? 'Checking...' : 'Check Proxy'}
				</button>
			{/if}
		</div>

		<!-- ============================================================= -->
		<!-- Proxy Status -->
		<!-- ============================================================= -->
		{#if proxyStatus}
			<div style="padding: 0.75rem; border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-default); margin-bottom: 1.25rem;">
				<div style="font-size: 0.8rem; font-weight: 600; color: var(--oo-fg-primary); margin-bottom: 0.5rem;">
					Proxy Status
				</div>
				<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.4rem; font-size: 0.8rem;">
					<div>
						<span style="color: var(--oo-fg-tertiary);">Configured:</span>
						<span style="color: var(--oo-fg-primary);">{proxyStatus.configured ? 'Yes' : 'No'}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-tertiary);">Reachable:</span>
						<span style="color: {proxyStatus.reachable ? 'var(--oo-status-success)' : 'var(--oo-status-error)'}; font-weight: 500;">
							{proxyStatus.reachable ? 'Yes' : 'No'}
						</span>
					</div>
					{#if proxyStatus.latency_ms !== null}
						<div>
							<span style="color: var(--oo-fg-tertiary);">Latency:</span>
							<span style="color: var(--oo-fg-primary);">{proxyStatus.latency_ms.toFixed(0)}ms</span>
						</div>
					{/if}
					{#if proxyStatus.exit_ip}
						<div>
							<span style="color: var(--oo-fg-tertiary);">Exit IP:</span>
							<span style="color: var(--oo-fg-primary); font-family: monospace; font-size: 0.75rem;">{proxyStatus.exit_ip}</span>
						</div>
					{/if}
					{#if proxyStatus.error}
						<div style="grid-column: 1 / -1;">
							<span style="color: var(--oo-status-error);">{proxyStatus.error}</span>
						</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- ============================================================= -->
		<!-- PII Sanitization -->
		<!-- ============================================================= -->
		<div style="margin-bottom: 1.25rem; padding-top: 0.75rem; border-top: 1px solid var(--oo-bd-default);">
			<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
				<label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer; font-size: 0.85rem; color: var(--oo-fg-primary); font-weight: 500;">
					<input
						type="checkbox"
						bind:checked={localPiiEnabled}
						style="accent-color: var(--oo-accent-primary);"
					/>
					PII Sanitization
				</label>
				<span style="font-size: 0.75rem; color: var(--oo-fg-tertiary);">
					Strip personal info from search queries
				</span>
			</div>

			{#if localPiiEnabled}
				<div style="margin-bottom: 0.5rem;">
					<label style="display: block; color: var(--oo-fg-secondary); font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">
						Test query
					</label>
					<div style="display: flex; gap: 0.4rem;">
						<input
							type="text"
							bind:value={piiTestQuery}
							placeholder="Enter a query to preview sanitization..."
							style="flex: 1; padding: 0.4rem 0.6rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-primary); font-size: 0.8rem;"
						/>
						<button
							style="padding: 0.4rem 0.7rem; border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default); cursor: pointer; font-size: 0.8rem; white-space: nowrap; opacity: {previewingPII ? 0.6 : 1};"
							on:click={runPIIPreview}
							disabled={previewingPII}
						>
							{previewingPII ? '...' : 'Preview'}
						</button>
					</div>
				</div>

				{#if piiPreview}
					<div style="padding: 0.6rem; border-radius: var(--oo-radius-md); background: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-default); font-size: 0.8rem;">
						<div style="margin-bottom: 0.4rem;">
							<span style="color: var(--oo-fg-tertiary);">Sanitized:</span>
							<span style="color: var(--oo-fg-primary); font-family: monospace; font-size: 0.75rem;">
								{piiPreview.sanitized}
							</span>
						</div>
						{#if piiPreview.items.length > 0}
							<div style="color: var(--oo-fg-tertiary); font-size: 0.75rem;">
								{#each piiPreview.items as item}
									<div style="margin-top: 0.2rem;">
										<span style="color: var(--oo-status-warning); font-weight: 500;">[{item.category}]</span>
										<span style="text-decoration: line-through; color: var(--oo-status-error);">{item.original}</span>
										&rarr;
										<span style="color: var(--oo-status-success);">{item.replacement}</span>
									</div>
								{/each}
							</div>
						{:else}
							<div style="color: var(--oo-fg-tertiary);">No PII detected.</div>
						{/if}
					</div>
				{/if}
			{/if}
		</div>

		<!-- ============================================================= -->
		<!-- Stats -->
		<!-- ============================================================= -->
		{#if searchConfig}
			<div style="padding-top: 0.75rem; border-top: 1px solid var(--oo-bd-default);">
				<div style="font-size: 0.8rem; font-weight: 600; color: var(--oo-fg-primary); margin-bottom: 0.5rem;">
					Search Stats
				</div>
				<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem; font-size: 0.8rem;">
					<div>
						<span style="color: var(--oo-fg-tertiary);">Total:</span>
						<span style="color: var(--oo-fg-primary);">{searchConfig.total_searches}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-tertiary);">Cache hits:</span>
						<span style="color: var(--oo-fg-primary);">{searchConfig.cache_hits}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-tertiary);">Errors:</span>
						<span style="color: {searchConfig.errors > 0 ? 'var(--oo-status-error)' : 'var(--oo-fg-primary)'};">{searchConfig.errors}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-tertiary);">Retries:</span>
						<span style="color: var(--oo-fg-primary);">{searchConfig.retries}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-tertiary);">PII stripped:</span>
						<span style="color: var(--oo-fg-primary);">{searchConfig.pii_sanitizations}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-tertiary);">Via proxy:</span>
						<span style="color: var(--oo-fg-primary);">{searchConfig.proxy_searches}</span>
					</div>
				</div>
				<div style="margin-top: 0.4rem; font-size: 0.75rem; color: var(--oo-fg-tertiary);">
					DDG: {searchConfig.ddgs_available ? 'available' : 'unavailable'}
					| PII module: {searchConfig.pii_available ? 'available' : 'unavailable'}
					| Cached: {searchConfig.cache_size}
				</div>
			</div>
		{/if}
	{/if}
</div>
