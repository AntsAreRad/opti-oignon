<!--
  SearchKillSwitchPanel.svelte
  Web search kill switch management panel.

  Features:
  - Status display: engaged/disengaged, circuit breaker state, injection count
  - One-click "Engage Kill Switch" button
  - Re-enable ceremony UI (visual code + password + 2FA + cooldown timer)
  - In Bulbe mode: re-enable disabled with explanation
  - Domain allowlist management: add/remove domains
  - Circuit breaker dashboard
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    getKillSwitchStatus,
    engageKillSwitch,
    requestReenable,
    getReenableCode,
    confirmReenable,
    cancelReenable,
    updateDomainAllowlist,
  } from '$lib/api/searchKillSwitch';
  import type { KillSwitchStatus } from '$lib/api/searchKillSwitch';
  import { isBulbe } from '$lib/stores/securityMode';

  let status: KillSwitchStatus = { available: false, search_enabled: true };
  let loading = true;
  let error = '';
  let actionLoading = false;

  // Re-enable ceremony state
  let reenableStep: 'idle' | 'pending' | 'confirm' = 'idle';
  let cooldownRemaining = 0;
  let visualCode = '';
  let pendingRequestId = '';
  let confirmCode = '';
  let confirmPassword = '';
  let confirmTwoFa = '';
  let confirmError = '';
  let cooldownInterval: ReturnType<typeof setInterval> | null = null;

  // Domain allowlist state
  let showDomains = false;
  let newDomain = '';
  let domainError = '';
  let domainSaving = false;
  let allowlistEnabled = false;
  let domainList: string[] = [];

  // Kill reason input
  let killReason = '';

  onMount(async () => {
    await loadStatus();
    loading = false;
  });

  onDestroy(() => {
    if (cooldownInterval) clearInterval(cooldownInterval);
  });

  async function loadStatus() {
    try {
      status = await getKillSwitchStatus();
      if (status.domain_allowlist) {
        allowlistEnabled = status.domain_allowlist.enabled;
        domainList = [...status.domain_allowlist.domains];
      }
      if (status.reenable_pending) {
        reenableStep = 'pending';
        await fetchVisualCode();
      }
    } catch (e: any) {
      error = e.message || 'Failed to load kill switch status';
    }
  }

  async function handleEngageKillSwitch() {
    actionLoading = true;
    error = '';
    try {
      const result = await engageKillSwitch(killReason || 'manual');
      if (result.success) {
        killReason = '';
        await loadStatus();
      } else {
        error = result.message || 'Failed to engage kill switch';
      }
    } catch (e: any) {
      error = e.message || 'Failed to engage kill switch';
    } finally {
      actionLoading = false;
    }
  }

  async function handleRequestReenable() {
    actionLoading = true;
    error = '';
    try {
      const result = await requestReenable();
      if (result.success && result.pending) {
        pendingRequestId = result.request_id || '';
        cooldownRemaining = result.cooldown_seconds || 300;
        reenableStep = 'pending';
        startCooldownTimer();
        await fetchVisualCode();
      } else {
        error = result.message || 'Re-enable request failed';
      }
    } catch (e: any) {
      error = e.message || 'Re-enable request failed';
    } finally {
      actionLoading = false;
    }
  }

  function startCooldownTimer() {
    if (cooldownInterval) clearInterval(cooldownInterval);
    cooldownInterval = setInterval(() => {
      cooldownRemaining = Math.max(0, cooldownRemaining - 1);
      if (cooldownRemaining <= 0) {
        if (cooldownInterval) clearInterval(cooldownInterval);
        reenableStep = 'confirm';
      }
    }, 1000);
  }

  async function fetchVisualCode() {
    try {
      const resp = await getReenableCode();
      visualCode = resp.visual_code;
    } catch (_) {
      visualCode = '';
    }
  }

  async function handleConfirmReenable() {
    actionLoading = true;
    confirmError = '';
    try {
      const result = await confirmReenable({
        request_id: pendingRequestId,
        visual_code: confirmCode,
        password: confirmPassword,
        two_fa_code: confirmTwoFa || null,
      });
      if (result.success) {
        resetReenableState();
        await loadStatus();
      } else {
        confirmError = result.message || 'Confirmation failed';
      }
    } catch (e: any) {
      confirmError = e.message || 'Confirmation failed';
    } finally {
      actionLoading = false;
    }
  }

  async function handleCancelReenable() {
    try {
      await cancelReenable();
    } catch (_) { /* ignore */ }
    resetReenableState();
    await loadStatus();
  }

  function resetReenableState() {
    reenableStep = 'idle';
    cooldownRemaining = 0;
    visualCode = '';
    confirmCode = '';
    confirmPassword = '';
    confirmTwoFa = '';
    pendingRequestId = '';
    confirmError = '';
    if (cooldownInterval) clearInterval(cooldownInterval);
  }

  // Domain allowlist management
  async function handleAddDomain() {
    const domain = newDomain.trim().toLowerCase();
    if (!domain) return;
    if (domainList.includes(domain)) {
      domainError = 'Domain already in list';
      return;
    }
    domainError = '';
    const updated = [...domainList, domain];
    await saveDomainAllowlist(allowlistEnabled, updated);
    newDomain = '';
  }

  async function handleRemoveDomain(domain: string) {
    const updated = domainList.filter((d) => d !== domain);
    await saveDomainAllowlist(allowlistEnabled, updated);
  }

  async function handleToggleAllowlist() {
    const newEnabled = !allowlistEnabled;
    await saveDomainAllowlist(newEnabled, domainList);
  }

  async function saveDomainAllowlist(enabled: boolean, domains: string[]) {
    domainSaving = true;
    domainError = '';
    try {
      const result = await updateDomainAllowlist(enabled, domains);
      if (result.success) {
        allowlistEnabled = result.enabled;
        domainList = [...result.domains];
      }
    } catch (e: any) {
      domainError = e.message || 'Failed to update domain allowlist';
    } finally {
      domainSaving = false;
    }
  }

  function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  function formatTimestamp(ts: number | null | undefined): string {
    if (!ts) return '-';
    return new Date(ts * 1000).toLocaleString();
  }

  $: isKilled = !status.search_enabled;
  $: circuitTripped = status.circuit_breaker_tripped ?? false;
  $: injectionCount = status.injection_count ?? 0;
  $: bulbeMode = $isBulbe;
</script>

{#if loading}
  <div class="flex items-center gap-2 p-4" style="color: var(--oo-fg-muted);">
    Loading kill switch status...
  </div>
{:else if !status.available}
  <div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <p class="text-sm" style="color: var(--oo-fg-muted);">Search kill switch module not available.</p>
  </div>
{:else}
  <!-- Status Card -->
  <div class="rounded-lg p-5 mb-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">
        Web Search Kill Switch
      </h3>
      <span
        class="text-xs px-2 py-1 rounded font-medium"
        style="background-color: {isKilled ? 'var(--oo-fg-error)' : 'var(--oo-sage)'}; color: white;"
      >
        {isKilled ? 'KILLED' : 'ACTIVE'}
      </span>
    </div>

    <!-- Status Summary -->
    <div class="grid grid-cols-3 gap-3 mb-4">
      <div class="rounded p-3 text-center" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Search</div>
        <div class="text-sm font-semibold" style="color: {isKilled ? 'var(--oo-fg-error)' : 'var(--oo-sage)'};">
          {isKilled ? 'Disabled' : 'Enabled'}
        </div>
      </div>
      <div class="rounded p-3 text-center" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Circuit Breaker</div>
        <div class="text-sm font-semibold" style="color: {circuitTripped ? 'var(--oo-fg-error)' : 'var(--oo-fg-primary)'};">
          {circuitTripped ? 'TRIPPED' : 'OK'}
        </div>
      </div>
      <div class="rounded p-3 text-center" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Injections</div>
        <div class="text-sm font-semibold" style="color: {injectionCount > 0 ? 'var(--oo-fg-warning)' : 'var(--oo-fg-primary)'};">
          {injectionCount}
        </div>
      </div>
    </div>

    <!-- Kill details if engaged -->
    {#if isKilled && status.killed_at}
      <div class="rounded p-3 mb-4 text-xs" style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-muted);">
        <div class="flex justify-between mb-1">
          <span>Killed at:</span>
          <span style="color: var(--oo-fg-primary);">{formatTimestamp(status.killed_at)}</span>
        </div>
        {#if status.killed_by}
          <div class="flex justify-between mb-1">
            <span>Killed by:</span>
            <span style="color: var(--oo-fg-primary);">{status.killed_by}</span>
          </div>
        {/if}
        {#if status.kill_reason}
          <div class="flex justify-between">
            <span>Reason:</span>
            <span style="color: var(--oo-fg-primary);">{status.kill_reason}</span>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Error -->
    {#if error}
      <div class="rounded p-2 mb-3 text-sm" style="background-color: var(--oo-bg-error); color: var(--oo-fg-error);">
        {error}
      </div>
    {/if}

    <!-- Action Buttons -->
    {#if reenableStep === 'idle'}
      <div class="flex flex-col gap-2">
        {#if !isKilled}
          <!-- Engage Kill Switch -->
          <div class="flex gap-2">
            <input
              type="text"
              class="flex-1 px-3 py-2 rounded text-sm"
              style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
              placeholder="Reason (optional)"
              aria-label="Kill switch reason"
              bind:value={killReason}
            />
            <button
              class="px-4 py-2 rounded text-sm font-medium whitespace-nowrap"
              style="background-color: var(--oo-fg-error); color: white;"
              on:click={handleEngageKillSwitch}
              disabled={actionLoading}
            >
              {actionLoading ? 'Engaging...' : 'Engage Kill Switch'}
            </button>
          </div>
        {:else}
          <!-- Re-enable button -->
          {#if bulbeMode}
            <div class="rounded p-3 text-sm" style="background-color: var(--oo-bg-subtle); border: 1px solid var(--oo-bd-subtle);">
              <p class="font-medium mb-1" style="color: var(--oo-fg-error);">
                Re-enable blocked
              </p>
              <p class="text-xs" style="color: var(--oo-fg-muted);">
                Web search cannot be re-enabled in Bulbe mode. This is a hardcoded
                restriction. Switch to Daily mode first if you need to restore search.
              </p>
            </div>
          {:else}
            <button
              class="px-4 py-2 rounded text-sm font-medium transition-colors"
              style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
              on:click={handleRequestReenable}
              disabled={actionLoading}
            >
              {actionLoading ? 'Requesting...' : 'Request Re-enable (Ceremony)'}
            </button>
          {/if}
        {/if}
      </div>
    {/if}

    <!-- Re-enable Ceremony: Pending/Cooldown -->
    {#if reenableStep === 'pending' || reenableStep === 'confirm'}
      <div class="rounded-lg p-4 mt-3" style="background-color: var(--oo-bg-warning); border: 1px solid var(--oo-fg-warning);">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-semibold" style="color: var(--oo-fg-warning);">
            Search re-enable ceremony
          </span>
          {#if cooldownRemaining > 0}
            <span class="text-sm font-mono" style="color: var(--oo-fg-warning);">
              {formatTime(cooldownRemaining)}
            </span>
          {/if}
        </div>

        {#if cooldownRemaining > 0}
          <p class="text-xs mb-2" style="color: var(--oo-fg-muted);">
            Cooldown active. You can cancel anytime.
          </p>
          <!-- Visual code (DOM-only, human-readable) -->
          {#if visualCode}
            <template data-security-code={visualCode}></template>
            <div class="visual-code-display mt-2 mb-2 text-center select-none">
              <span class="text-2xl font-mono tracking-widest" style="
                color: var(--oo-fg-primary);
                background: var(--oo-bg-subtle);
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                letter-spacing: 0.3em;
                font-family: 'Courier New', monospace;
              ">
                {visualCode}
              </span>
            </div>
            <p class="text-xs text-center" style="color: var(--oo-fg-muted);">
              You will need this code to confirm re-enabling search.
            </p>
          {/if}
          <button
            class="mt-2 text-xs underline"
            style="color: var(--oo-fg-muted);"
            on:click={handleCancelReenable}
          >
            Cancel re-enable
          </button>
        {:else}
          <!-- Confirmation form -->
          <p class="text-xs mb-3" style="color: var(--oo-fg-muted);">
            Cooldown complete. Enter the visual code and your credentials.
          </p>
          <div class="space-y-2">
            <input
              type="text"
              class="w-full px-3 py-2 rounded text-sm"
              style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
              placeholder="6-digit confirmation code"
              aria-label="Six-digit confirmation code"
              maxlength="6"
              bind:value={confirmCode}
            />
            <input
              type="password"
              class="w-full px-3 py-2 rounded text-sm"
              style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
              placeholder="Current password"
              aria-label="Current password"
              bind:value={confirmPassword}
            />
            <input
              type="text"
              class="w-full px-3 py-2 rounded text-sm"
              style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
              placeholder="2FA code (if enabled)"
              aria-label="Two-factor authentication code"
              bind:value={confirmTwoFa}
            />
            {#if confirmError}
              <p class="text-xs" style="color: var(--oo-fg-error);">{confirmError}</p>
            {/if}
            <div class="flex gap-2">
              <button
                class="px-4 py-2 rounded text-sm font-medium"
                style="background-color: var(--oo-fg-warning); color: white;"
                on:click={handleConfirmReenable}
                disabled={actionLoading || !confirmCode || !confirmPassword}
              >
                {actionLoading ? 'Confirming...' : 'Confirm Re-enable'}
              </button>
              <button
                class="px-4 py-2 rounded text-sm"
                style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
                on:click={handleCancelReenable}
              >
                Cancel
              </button>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Domain Allowlist Card -->
  <div class="rounded-lg p-5 mb-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">
        Domain Allowlist
      </h3>
      <div class="flex items-center gap-2">
        <span class="text-xs" style="color: var(--oo-fg-muted);">
          {allowlistEnabled ? 'Enforced' : 'Off'}
        </span>
        <button
          class="relative w-9 h-5 rounded-full transition-colors"
          style="background-color: {allowlistEnabled ? 'var(--oo-sage)' : 'var(--oo-bg-subtle)'}; border: 1px solid var(--oo-bd-subtle);"
          on:click={handleToggleAllowlist}
          disabled={domainSaving}
          title={allowlistEnabled ? 'Disable domain allowlist' : 'Enable domain allowlist'}
          aria-label={allowlistEnabled ? 'Disable domain allowlist' : 'Enable domain allowlist'}
        >
          <span
            class="absolute top-0.5 rounded-full w-4 h-4 transition-transform"
            style="background-color: white; left: {allowlistEnabled ? '1rem' : '0.125rem'};"
          ></span>
        </button>
      </div>
    </div>

    <p class="text-xs mb-3" style="color: var(--oo-fg-muted);">
      When enabled, search results are filtered to only include domains in this list.
      Server-enforced: the LLM never sees results from unlisted domains.
    </p>

    {#if domainError}
      <div class="rounded p-2 mb-2 text-xs" style="background-color: var(--oo-bg-error); color: var(--oo-fg-error);">
        {domainError}
      </div>
    {/if}

    <!-- Add domain -->
    <div class="flex gap-2 mb-3">
      <input
        type="text"
        class="flex-1 px-3 py-2 rounded text-sm"
        style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
        placeholder="example.com"
        aria-label="Domain to add to allowlist"
        bind:value={newDomain}
        on:keydown={(e) => { if (e.key === 'Enter') handleAddDomain(); }}
      />
      <button
        class="px-3 py-2 rounded text-sm font-medium"
        style="background-color: var(--oo-sage); color: white;"
        on:click={handleAddDomain}
        disabled={domainSaving || !newDomain.trim()}
      >
        Add
      </button>
    </div>

    <!-- Domain list -->
    <button
      class="text-xs mb-2"
      style="color: var(--oo-fg-muted);"
      on:click={() => showDomains = !showDomains}
    >
      {showDomains ? 'Hide' : 'Show'} domains ({domainList.length})
    </button>

    {#if showDomains}
      {#if domainList.length === 0}
        <p class="text-xs py-2" style="color: var(--oo-fg-muted);">No domains configured.</p>
      {:else}
        <div class="space-y-1 max-h-48 overflow-y-auto">
          {#each domainList as domain}
            <div
              class="flex items-center justify-between px-3 py-1.5 rounded text-sm"
              style="background-color: var(--oo-bg-subtle);"
            >
              <span class="font-mono text-xs" style="color: var(--oo-fg-primary);">{domain}</span>
              <button
                class="text-xs px-2 py-0.5 rounded"
                style="color: var(--oo-fg-error); border: 1px solid var(--oo-fg-error);"
                on:click={() => handleRemoveDomain(domain)}
                disabled={domainSaving}
              >
                Remove
              </button>
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  </div>

  <!-- Circuit Breaker Dashboard Card -->
  <div class="rounded-lg p-5" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
    <h3 class="text-base font-semibold mb-3" style="color: var(--oo-fg-primary);">
      Circuit Breaker
    </h3>

    <p class="text-xs mb-3" style="color: var(--oo-fg-muted);">
      Automatically disables search if 3 injection attempts are detected within
      10 minutes. This is a server-side protection that operates independently
      of the kill switch.
    </p>

    <div class="flex gap-3">
      <div class="flex-1 rounded p-3" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Status</div>
        <div class="text-sm font-semibold" style="color: {circuitTripped ? 'var(--oo-fg-error)' : 'var(--oo-sage)'};">
          {circuitTripped ? 'TRIPPED' : 'Normal'}
        </div>
      </div>
      <div class="flex-1 rounded p-3" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Injection Count</div>
        <div class="text-sm font-semibold" style="color: {injectionCount > 0 ? 'var(--oo-fg-warning)' : 'var(--oo-fg-primary)'};">
          {injectionCount}
        </div>
      </div>
      <div class="flex-1 rounded p-3" style="background-color: var(--oo-bg-subtle);">
        <div class="text-xs mb-1" style="color: var(--oo-fg-muted);">Threshold</div>
        <div class="text-sm font-semibold" style="color: var(--oo-fg-primary);">
          3 / 10 min
        </div>
      </div>
    </div>

    {#if circuitTripped}
      <div class="rounded p-3 mt-3 text-xs" style="background-color: var(--oo-bg-error); color: var(--oo-fg-error);">
        Circuit breaker tripped: search has been auto-disabled due to detected
        injection attempts. The kill switch must be explicitly re-enabled through
        the ceremony to restore search functionality.
      </div>
    {/if}
  </div>
{/if}

<style>
  .visual-code-display {
    user-select: none;
    -webkit-user-select: none;
  }
</style>
