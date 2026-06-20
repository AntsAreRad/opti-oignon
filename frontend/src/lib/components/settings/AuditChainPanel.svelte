<!--
  AuditChainPanel.svelte (S130)
  Hash-chain signed audit log viewer.

  Features:
    - Chain status: length, last entry, integrity badge
    - "Verify Chain" button with progress indicator
    - Event table: paginated, filterable by event_type and severity
    - Expandable event detail rows
    - CSV export button
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getAuditChainStatus,
		getAuditChainEvents,
		verifyAuditChain,
		exportAuditChainCsv,
		type AuditChainStatus,
		type AuditEvent,
		type AuditChainVerifyResult,
	} from '../../api/auditChain';

	// -- State ---------------------------------------------------------------

	let status: AuditChainStatus | null = null;
	let events: AuditEvent[] = [];
	let loading = true;
	let error = '';

	// Pagination
	let currentPage = 0;
	const pageSize = 20;
	let hasMore = false;

	// Filters
	let filterEventType = '';
	let filterSeverity = '';

	// Verification
	let verifying = false;
	let verifyResult: AuditChainVerifyResult | null = null;

	// Export
	let exporting = false;

	// Expanded rows
	let expandedRows: Set<number> = new Set();

	// -- Lifecycle ------------------------------------------------------------

	onMount(async () => {
		await loadAll();
		loading = false;
	});

	async function loadAll() {
		try {
			status = await getAuditChainStatus();
			await loadEvents();
			error = '';
		} catch (e: any) {
			error = e?.message || 'Failed to load audit chain';
		}
	}

	async function loadEvents() {
		try {
			const params: Record<string, any> = {
				limit: pageSize,
				offset: currentPage * pageSize,
			};
			if (filterEventType) params.event_type = filterEventType;
			if (filterSeverity) params.severity = filterSeverity;

			const resp = await getAuditChainEvents(params);
			events = resp.events;
			hasMore = resp.count >= pageSize;
		} catch (e: any) {
			error = e?.message || 'Failed to load events';
		}
	}

	async function handleVerify() {
		verifying = true;
		verifyResult = null;
		try {
			verifyResult = await verifyAuditChain();
			// Refresh status after verify
			status = await getAuditChainStatus();
		} catch (e: any) {
			error = e?.message || 'Verification failed';
		} finally {
			verifying = false;
		}
	}

	async function handleExport() {
		exporting = true;
		try {
			const csv = await exportAuditChainCsv();
			const blob = new Blob([csv], { type: 'text/csv' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'audit_chain.csv';
			a.click();
			URL.revokeObjectURL(url);
		} catch (e: any) {
			error = e?.message || 'Export failed';
		} finally {
			exporting = false;
		}
	}

	async function handleFilter() {
		currentPage = 0;
		await loadEvents();
	}

	async function prevPage() {
		if (currentPage > 0) {
			currentPage--;
			await loadEvents();
		}
	}

	async function nextPage() {
		if (hasMore) {
			currentPage++;
			await loadEvents();
		}
	}

	function toggleRow(id: number) {
		if (expandedRows.has(id)) {
			expandedRows.delete(id);
		} else {
			expandedRows.add(id);
		}
		expandedRows = expandedRows;
	}

	function formatTimestamp(ts: number): string {
		return new Date(ts * 1000).toLocaleString();
	}

	function severityColor(sev: string): string {
		switch (sev) {
			case 'CRITICAL': return 'var(--oo-fg-error)';
			case 'WARNING': return 'var(--oo-fg-warning)';
			default: return 'var(--oo-fg-muted)';
		}
	}

	function truncateHash(hash: string): string {
		if (!hash || hash.length < 16) return hash || '';
		return hash.slice(0, 8) + '...' + hash.slice(-8);
	}
</script>

<div class="space-y-4">
	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading audit chain...</p>

	{:else if error}
		<div class="rounded-lg p-3" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-fg-error);">
			<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
		</div>

	{:else}
		<!-- Chain Status Card -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center justify-between mb-3">
				<h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">Chain Status</h3>
				<div class="flex items-center gap-2">
					{#if status?.chain_valid}
						<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
							style="background-color: var(--oo-sage); color: var(--oo-bg-primary);">
							&#10003; Intact
						</span>
					{:else}
						<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
							style="background-color: var(--oo-fg-error); color: var(--oo-bg-primary);">
							&#10007; Broken
						</span>
					{/if}
				</div>
			</div>

			<div class="grid grid-cols-3 gap-4 text-sm">
				<div>
					<span style="color: var(--oo-fg-muted);">Entries</span>
					<p class="font-mono font-semibold" style="color: var(--oo-fg-primary);">
						{status?.total_entries ?? 0}
					</p>
				</div>
				<div>
					<span style="color: var(--oo-fg-muted);">Last Entry</span>
					<p class="font-mono text-xs" style="color: var(--oo-fg-primary);">
						{status?.last_entry ? formatTimestamp(status.last_entry.timestamp) : 'None'}
					</p>
				</div>
				<div>
					<span style="color: var(--oo-fg-muted);">Last Hash</span>
					<p class="font-mono text-xs" style="color: var(--oo-fg-secondary);">
						{status?.last_entry ? truncateHash(status.last_entry.entry_hash) : 'N/A'}
					</p>
				</div>
			</div>

			{#if status && !status.chain_valid && status.first_broken_index !== null}
				<p class="mt-2 text-xs" style="color: var(--oo-fg-error);">
					Chain integrity broken at entry #{status.first_broken_index}. Investigate immediately.
				</p>
			{/if}
		</div>

		<!-- Actions Row -->
		<div class="flex gap-2 flex-wrap">
			<button
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-tobacco); color: var(--oo-bg-primary);"
				on:click={handleVerify}
				disabled={verifying}
			>
				{#if verifying}
					Verifying...
				{:else}
					Verify Chain
				{/if}
			</button>

			<button
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
				on:click={handleExport}
				disabled={exporting}
			>
				{exporting ? 'Exporting...' : 'Export CSV'}
			</button>
		</div>

		<!-- Verify Result -->
		{#if verifyResult}
			<div class="rounded-lg p-3 text-sm"
				style="background-color: var(--oo-card-bg); border: 1px solid {verifyResult.chain_valid ? 'var(--oo-sage)' : 'var(--oo-fg-error)'};">
				{#if verifyResult.chain_valid}
					<span style="color: var(--oo-sage);">&#10003;</span>
					Chain verified: {verifyResult.total_entries} entries, all hashes valid.
				{:else}
					<span style="color: var(--oo-fg-error);">&#10007;</span>
					Chain broken at entry #{verifyResult.first_broken_index}
					({verifyResult.total_entries} total entries).
				{/if}
			</div>
		{/if}

		<!-- Filters -->
		<div class="flex gap-2 items-end flex-wrap">
			<div>
				<label class="block text-xs mb-1" style="color: var(--oo-fg-muted);">Event Type</label>
				<input
					type="text"
					class="px-2 py-1 rounded text-xs"
					style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
					placeholder="e.g. login_success"
					aria-label="Filter audit events by type"
					bind:value={filterEventType}
				/>
			</div>
			<div>
				<label class="block text-xs mb-1" style="color: var(--oo-fg-muted);">Severity</label>
				<select
					class="px-2 py-1 rounded text-xs"
					style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
					bind:value={filterSeverity}
				>
					<option value="">All</option>
					<option value="INFO">INFO</option>
					<option value="WARNING">WARNING</option>
					<option value="CRITICAL">CRITICAL</option>
				</select>
			</div>
			<button
				class="px-3 py-1 rounded text-xs font-medium"
				style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
				on:click={handleFilter}
			>
				Filter
			</button>
		</div>

		<!-- Events Table -->
		<div class="rounded-lg overflow-hidden" style="border: 1px solid var(--oo-bd-subtle);">
			<table class="w-full text-xs">
				<thead>
					<tr style="background-color: var(--oo-bg-subtle);">
						<th class="text-left px-3 py-2 font-medium" style="color: var(--oo-fg-muted);">#</th>
						<th class="text-left px-3 py-2 font-medium" style="color: var(--oo-fg-muted);">Time</th>
						<th class="text-left px-3 py-2 font-medium" style="color: var(--oo-fg-muted);">Event</th>
						<th class="text-left px-3 py-2 font-medium" style="color: var(--oo-fg-muted);">Source</th>
						<th class="text-left px-3 py-2 font-medium" style="color: var(--oo-fg-muted);">Severity</th>
						<th class="text-left px-3 py-2 font-medium" style="color: var(--oo-fg-muted);">Hash</th>
					</tr>
				</thead>
				<tbody>
					{#if events.length === 0}
						<tr>
							<td colspan="6" class="px-3 py-4 text-center" style="color: var(--oo-fg-muted);">
								No audit events found.
							</td>
						</tr>
					{:else}
						{#each events as evt}
							<tr
								class="cursor-pointer transition-colors"
								style="border-top: 1px solid var(--oo-bd-subtle); background-color: var(--oo-card-bg);"
								on:click={() => toggleRow(evt.id)}
							>
								<td class="px-3 py-2 font-mono" style="color: var(--oo-fg-secondary);">{evt.id}</td>
								<td class="px-3 py-2" style="color: var(--oo-fg-secondary);">{formatTimestamp(evt.timestamp)}</td>
								<td class="px-3 py-2 font-mono" style="color: var(--oo-fg-primary);">{evt.event_type}</td>
								<td class="px-3 py-2" style="color: var(--oo-fg-secondary);">{evt.source}</td>
								<td class="px-3 py-2 font-medium" style="color: {severityColor(evt.severity)};">{evt.severity}</td>
								<td class="px-3 py-2 font-mono" style="color: var(--oo-fg-muted);">{truncateHash(evt.entry_hash)}</td>
							</tr>
							<!-- Expanded detail row -->
							{#if expandedRows.has(evt.id)}
								<tr style="border-top: 1px solid var(--oo-bd-subtle); background-color: var(--oo-bg-subtle);">
									<td colspan="6" class="px-4 py-3">
										<div class="space-y-2 text-xs">
											<div>
												<span class="font-medium" style="color: var(--oo-fg-muted);">Action:</span>
												<span style="color: var(--oo-fg-primary);">{evt.action}</span>
											</div>
											<div>
												<span class="font-medium" style="color: var(--oo-fg-muted);">Details:</span>
												<pre class="mt-1 p-2 rounded font-mono overflow-x-auto"
													style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary);"
												>{JSON.stringify(evt.details, null, 2)}</pre>
											</div>
											<div>
												<span class="font-medium" style="color: var(--oo-fg-muted);">Entry Hash:</span>
												<span class="font-mono break-all" style="color: var(--oo-fg-secondary);">{evt.entry_hash}</span>
											</div>
											<div>
												<span class="font-medium" style="color: var(--oo-fg-muted);">Prev Hash:</span>
												<span class="font-mono break-all" style="color: var(--oo-fg-secondary);">{evt.prev_hash}</span>
											</div>
										</div>
									</td>
								</tr>
							{/if}
						{/each}
					{/if}
				</tbody>
			</table>
		</div>

		<!-- Pagination -->
		{#if events.length > 0}
			<div class="flex items-center justify-between text-xs" style="color: var(--oo-fg-muted);">
				<span>Page {currentPage + 1}</span>
				<div class="flex gap-2">
					<button
						class="px-2 py-1 rounded"
						style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-subtle); color: var(--oo-fg-primary);"
						on:click={prevPage}
						disabled={currentPage === 0}
					>
						&#8592; Prev
					</button>
					<button
						class="px-2 py-1 rounded"
						style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-subtle); color: var(--oo-fg-primary);"
						on:click={nextPage}
						disabled={!hasMore}
					>
						Next &#8594;
					</button>
				</div>
			</div>
		{/if}
	{/if}
</div>
