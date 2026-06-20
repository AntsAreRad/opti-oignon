<!--
  PluginAllowlistPanel.svelte (S128)
  Plugin batch approval and allowlist management panel.

  Features:
  - Approved plugins table with hash, date, batch, permissions
  - Batch approval ceremony (prepare -> review manifest -> confirm with creds)
  - Per-plugin revoke button
  - Batch revoke (revoke all from one ceremony)
  - Permission escalation warnings
  - Daily mode: info message (allowlist not enforced)
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    getAllowlistStatus,
    approveBatch,
    revokePlugin,
    revokeBatch,
  } from '$lib/api/pluginAllowlist';
  import type { AllowlistStatus, AllowlistEntry, BatchManifest } from '$lib/api/pluginAllowlist';
  import { isBulbe } from '$lib/stores/securityMode';

  let status: AllowlistStatus = {
    available: false,
    total_entries: 0,
    batches: {},
    pending_batch: null,
    entries: [],
  };
  let loading = true;
  let error = '';
  let actionLoading = false;

  // Batch approval ceremony state
  let approvalStep: 'idle' | 'review' | 'confirm' = 'idle';
  let pendingManifest: BatchManifest | null = null;
  let approvePassword = '';
  let approveTwoFa = '';
  let approveVisualCode = '';
  let approveError = '';

  // Revoke confirmation
  let revokeTarget: { type: 'plugin' | 'batch'; id: string } | null = null;
  let revokeLoading = false;

  // Expand/collapse
  let showEntries = true;

  onMount(async () => {
    await loadStatus();
    loading = false;
  });

  async function loadStatus() {
    try {
      status = await getAllowlistStatus();
      // Resume if there is a pending batch from server
      if (status.pending_batch) {
        pendingManifest = status.pending_batch;
        approvalStep = 'review';
      }
    } catch (e: any) {
      error = e.message || 'Failed to load plugin allowlist';
    }
  }

  async function handleApproveBatch() {
    if (!pendingManifest) return;
    actionLoading = true;
    approveError = '';
    try {
      const result = await approveBatch({
        batch_id: pendingManifest.batch_id,
        visual_code: approveVisualCode,
        password: approvePassword,
        two_fa_code: approveTwoFa || null,
      });
      if (result.success) {
        resetApprovalState();
        await loadStatus();
      } else {
        approveError = result.message || 'Batch approval failed';
      }
    } catch (e: any) {
      approveError = e.message || 'Batch approval failed';
    } finally {
      actionLoading = false;
    }
  }

  function resetApprovalState() {
    approvalStep = 'idle';
    pendingManifest = null;
    approvePassword = '';
    approveTwoFa = '';
    approveVisualCode = '';
    approveError = '';
  }

  async function handleRevokePlugin(pluginId: string) {
    revokeLoading = true;
    error = '';
    try {
      const result = await revokePlugin(pluginId);
      if (result.success) {
        revokeTarget = null;
        await loadStatus();
      }
    } catch (e: any) {
      error = e.message || 'Revoke failed';
    } finally {
      revokeLoading = false;
    }
  }

  async function handleRevokeBatch(batchId: string) {
    revokeLoading = true;
    error = '';
    try {
      const result = await revokeBatch(batchId);
      if (result.success) {
        revokeTarget = null;
        await loadStatus();
      }
    } catch (e: any) {
      error = e.message || 'Batch revoke failed';
    } finally {
      revokeLoading = false;
    }
  }

  function confirmRevoke(type: 'plugin' | 'batch', id: string) {
    revokeTarget = { type, id };
  }

  function cancelRevoke() {
    revokeTarget = null;
  }

  function formatTimestamp(ts: number): string {
    if (!ts) return '-';
    return new Date(ts * 1000).toLocaleString();
  }

  function truncateHash(hash: string): string {
    if (!hash) return '-';
    // "sha512:abcdef..." -> show prefix
    if (hash.length > 24) {
      return hash.substring(0, 24) + '...';
    }
    return hash;
  }

  // Group entries by batch for batch revoke
  function uniqueBatches(entries: AllowlistEntry[]): string[] {
    const seen = new Set<string>();
    for (const e of entries) {
      seen.add(e.batch_id);
    }
    return Array.from(seen);
  }

  $: bulbeMode = $isBulbe;
  $: entryCount = status.total_entries;
  $: batchIds = uniqueBatches(status.entries);
</script>

{#if loading}
  <div class="flex items-center gap-2 p-4" style="color: var(--oo-fg-muted);">
    Loading plugin allowlist...
  </div>
{:else if !status.available}
  <div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <p class="text-sm" style="color: var(--oo-fg-muted);">Plugin allowlist module not available.</p>
  </div>
{:else}
  <!-- Mode Info Banner -->
  {#if !bulbeMode}
    <div class="rounded-lg p-3 mb-4" style="background-color: var(--oo-bg-subtle); border: 1px solid var(--oo-bd-subtle);">
      <p class="text-sm" style="color: var(--oo-fg-muted);">
        <span class="font-medium" style="color: var(--oo-sage);">Daily mode</span> —
        Plugin allowlist is not enforced. All plugins load normally.
        The allowlist is only checked in Bulbe mode.
      </p>
    </div>
  {/if}

  <!-- Allowlist Status Card -->
  <div class="rounded-lg p-5 mb-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">
        Plugin Allowlist
      </h3>
      <span
        class="text-xs px-2 py-1 rounded font-medium"
        style="background-color: {bulbeMode ? 'var(--oo-fg-error)' : 'var(--oo-sage)'}; color: white;"
      >
        {bulbeMode ? 'ENFORCED' : 'INACTIVE'}
      </span>
    </div>

    <!-- Summary -->
    <div class="grid grid-cols-2 gap-3 mb-4">
      <div class="rounded p-3 text-center" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Approved Plugins</div>
        <div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">{entryCount}</div>
      </div>
      <div class="rounded p-3 text-center" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Approval Batches</div>
        <div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">{batchIds.length}</div>
      </div>
    </div>

    <!-- Error -->
    {#if error}
      <div class="rounded p-2 mb-3 text-sm" style="background-color: var(--oo-bg-error); color: var(--oo-fg-error);">
        {error}
      </div>
    {/if}

    <!-- Entries Table -->
    <div class="flex items-center justify-between mb-2">
      <button
        class="text-xs"
        style="color: var(--oo-fg-muted);"
        on:click={() => showEntries = !showEntries}
      >
        {showEntries ? 'Hide' : 'Show'} approved plugins
      </button>
    </div>

    {#if showEntries}
      {#if status.entries.length === 0}
        <p class="text-xs py-3" style="color: var(--oo-fg-muted);">
          No plugins approved yet.
        </p>
      {:else}
        <div class="rounded overflow-hidden" style="border: 1px solid var(--oo-bd-subtle);">
          <div class="overflow-x-auto">
            <table class="w-full text-xs">
              <thead>
                <tr style="background-color: var(--oo-bg-subtle);">
                  <th class="text-left p-2 font-medium" style="color: var(--oo-fg-secondary);">Plugin</th>
                  <th class="text-left p-2 font-medium" style="color: var(--oo-fg-secondary);">Hash</th>
                  <th class="text-left p-2 font-medium" style="color: var(--oo-fg-secondary);">Permissions</th>
                  <th class="text-left p-2 font-medium" style="color: var(--oo-fg-secondary);">Approved</th>
                  <th class="text-right p-2 font-medium" style="color: var(--oo-fg-secondary);">Action</th>
                </tr>
              </thead>
              <tbody>
                {#each status.entries as entry}
                  <tr style="border-top: 1px solid var(--oo-bd-subtle);">
                    <td class="p-2 font-medium" style="color: var(--oo-fg-primary);">
                      {entry.plugin_id}
                    </td>
                    <td class="p-2 font-mono" style="color: var(--oo-fg-muted);" title={entry.code_hash}>
                      {truncateHash(entry.code_hash)}
                    </td>
                    <td class="p-2" style="color: var(--oo-fg-muted);">
                      {#if entry.permissions.length === 0}
                        <span style="color: var(--oo-sage);">none</span>
                      {:else}
                        {entry.permissions.join(', ')}
                      {/if}
                    </td>
                    <td class="p-2 whitespace-nowrap" style="color: var(--oo-fg-muted);">
                      {formatTimestamp(entry.approved_at)}
                    </td>
                    <td class="p-2 text-right">
                      {#if revokeTarget?.type === 'plugin' && revokeTarget.id === entry.plugin_id}
                        <div class="flex justify-end gap-1">
                          <button
                            class="px-2 py-0.5 rounded text-xs"
                            style="background-color: var(--oo-fg-error); color: white;"
                            on:click={() => handleRevokePlugin(entry.plugin_id)}
                            disabled={revokeLoading}
                          >
                            {revokeLoading ? '...' : 'Confirm'}
                          </button>
                          <button
                            class="px-2 py-0.5 rounded text-xs"
                            style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
                            on:click={cancelRevoke}
                          >
                            No
                          </button>
                        </div>
                      {:else}
                        <button
                          class="px-2 py-0.5 rounded text-xs"
                          style="color: var(--oo-fg-error); border: 1px solid var(--oo-fg-error);"
                          on:click={() => confirmRevoke('plugin', entry.plugin_id)}
                        >
                          Revoke
                        </button>
                      {/if}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/if}
    {/if}

    <!-- Batch Revoke Section -->
    {#if batchIds.length > 0}
      <div class="mt-4">
        <h4 class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">
          Batch Revoke
        </h4>
        <div class="space-y-1">
          {#each batchIds as bId}
            <div
              class="flex items-center justify-between px-3 py-2 rounded text-xs"
              style="background-color: var(--oo-bg-subtle);"
            >
              <span class="font-mono" style="color: var(--oo-fg-primary);">
                {bId.length > 20 ? bId.substring(0, 20) + '...' : bId}
              </span>
              <span style="color: var(--oo-fg-muted);">
                {status.batches[bId] ?? 0} plugin{(status.batches[bId] ?? 0) !== 1 ? 's' : ''}
              </span>
              {#if revokeTarget?.type === 'batch' && revokeTarget.id === bId}
                <div class="flex gap-1">
                  <button
                    class="px-2 py-0.5 rounded text-xs"
                    style="background-color: var(--oo-fg-error); color: white;"
                    on:click={() => handleRevokeBatch(bId)}
                    disabled={revokeLoading}
                  >
                    {revokeLoading ? '...' : 'Confirm'}
                  </button>
                  <button
                    class="px-2 py-0.5 rounded text-xs"
                    style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
                    on:click={cancelRevoke}
                  >
                    No
                  </button>
                </div>
              {:else}
                <button
                  class="px-2 py-0.5 rounded text-xs"
                  style="color: var(--oo-fg-error); border: 1px solid var(--oo-fg-error);"
                  on:click={() => confirmRevoke('batch', bId)}
                >
                  Revoke All
                </button>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <!-- Batch Approval Ceremony Card -->
  {#if approvalStep === 'review' && pendingManifest}
    <div class="rounded-lg p-5 mb-4" style="background-color: var(--oo-card-bg); border: 2px solid var(--oo-fg-warning);">
      <h3 class="text-base font-semibold mb-3" style="color: var(--oo-fg-warning);">
        Batch Approval Ceremony
      </h3>

      <p class="text-xs mb-3" style="color: var(--oo-fg-muted);">
        Review the batch manifest below. Each plugin's SHA-512 hash has been
        computed from its source files. Confirm with your credentials to approve.
      </p>

      <!-- Batch Manifest -->
      <div class="rounded p-3 mb-4" style="background-color: var(--oo-bg-subtle); border: 1px solid var(--oo-bd-subtle);">
        <div class="text-xs mb-2" style="color: var(--oo-fg-muted);">
          Batch ID:
          <span class="font-mono" style="color: var(--oo-fg-primary);">{pendingManifest.batch_id}</span>
        </div>
        <div class="text-xs mb-3" style="color: var(--oo-fg-muted);">
          Batch hash:
          <span class="font-mono" style="color: var(--oo-fg-primary);">
            {truncateHash(pendingManifest.batch_hash)}
          </span>
        </div>

        <div class="space-y-2">
          {#each pendingManifest.plugins as plugin}
            <div class="rounded p-2" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
              <div class="font-medium text-sm mb-1" style="color: var(--oo-fg-primary);">
                {plugin.plugin_id}
              </div>
              <div class="text-xs font-mono mb-1" style="color: var(--oo-fg-muted);" title={plugin.code_hash}>
                {plugin.code_hash}
              </div>
              {#if plugin.permissions.length > 0}
                <div class="flex flex-wrap gap-1 mt-1">
                  {#each plugin.permissions as perm}
                    <span
                      class="text-xs px-1.5 py-0.5 rounded"
                      style="background-color: var(--oo-fg-warning); color: white;"
                    >
                      {perm}
                    </span>
                  {/each}
                </div>
              {:else}
                <span class="text-xs" style="color: var(--oo-sage);">No special permissions</span>
              {/if}
            </div>
          {/each}
        </div>
      </div>

      <!-- Ceremony Credentials -->
      <div class="space-y-2">
        <input
          type="text"
          class="w-full px-3 py-2 rounded text-sm"
          style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
          placeholder="Visual code (if ceremony was initiated)"
          aria-label="Visual confirmation code"
          bind:value={approveVisualCode}
        />
        <input
          type="password"
          class="w-full px-3 py-2 rounded text-sm"
          style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
          placeholder="Current password"
          aria-label="Current password"
          bind:value={approvePassword}
        />
        <input
          type="text"
          class="w-full px-3 py-2 rounded text-sm"
          style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
          placeholder="2FA code (if enabled)"
          aria-label="Two-factor authentication code"
          bind:value={approveTwoFa}
        />
        {#if approveError}
          <p class="text-xs" style="color: var(--oo-fg-error);">{approveError}</p>
        {/if}
        <div class="flex gap-2">
          <button
            class="px-4 py-2 rounded text-sm font-medium"
            style="background-color: var(--oo-fg-warning); color: white;"
            on:click={handleApproveBatch}
            disabled={actionLoading || !approvePassword}
          >
            {actionLoading ? 'Approving...' : 'Approve Batch'}
          </button>
          <button
            class="px-4 py-2 rounded text-sm"
            style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
            on:click={resetApprovalState}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Permission Escalation Info -->
  <div class="rounded-lg p-5" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <h3 class="text-base font-semibold mb-3" style="color: var(--oo-fg-primary);">
      Permission Escalation Detection
    </h3>
    <p class="text-xs" style="color: var(--oo-fg-muted);">
      In Bulbe mode, if a plugin requests permissions beyond what was approved,
      it is automatically blocked and requires re-approval. This protects against
      plugins that silently add new capabilities after initial approval.
    </p>
    <p class="text-xs mt-2" style="color: var(--oo-fg-muted);">
      Hash verification runs at every plugin load. If any source file changes
      after approval, the plugin is blocked until the batch is re-approved
      through a new ceremony.
    </p>
  </div>
{/if}
