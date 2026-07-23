<!--
  SandboxSettingsStrip.svelte (Sandbox Workspace cycle, Bloc 4)
  The per-workspace settings strip (spec sections 8 and 10): the
  per-sandbox command timeout and the configured resource caps shown
  read-only, and the NETWORK toggle -- the cycle's one new capability and
  its most sensitive. The flag is per workspace, default off, Daily-only,
  and an explicit USER action: the model can trigger nothing here. Under
  Bulbe the toggle is DISABLED and the refusal is stated honestly (the
  SyncPanel precedent; the server gate is fail-secure -- an unset or
  unknown mode is treated as Bulbe and the API answers 403). When the
  network is ON the strip says so plainly and warns that the workspace
  has a second exit the approval gate does NOT cover -- sharpened when
  host files were cloned in. The provision row (visible only when the
  network is on) runs the one scoped egress: a hash-pinned requirements
  set installed with --require-hashes --only-binary=:all: into a
  workspace venv; refusals are surfaced per line, honestly. Design-system
  tokens only (--oo-*); lucide icons through Icon. Registered in
  FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Icon, InlineError, Input, Switch } from '$lib/ds';
	import { setNetwork, provisionWorkspace } from '$lib/api/sandbox';
	import type {
		SandboxSessionInfo,
		SandboxStatusResponse,
		SandboxProvisionResponse
	} from '$lib/types';

	export let session: SandboxSessionInfo | null = null;
	export let status: SandboxStatusResponse | null = null;
	export let disabled = false;

	const dispatch = createEventDispatcher<{
		networkChanged: { sessionId: string; enabled: boolean };
	}>();

	let toggling = false;
	let provisioning = false;
	let error: string | null = null;
	let requirementsPath = 'requirements.txt';
	let venvDir = '.venv';
	let provisionResult: SandboxProvisionResponse | null = null;

	$: networkAllowed = status?.network_allowed ?? false;
	$: networkOn = session?.network_enabled ?? false;
	$: hasClonedBaseline = session?.has_cloned_baseline ?? false;
	$: effectiveTimeout =
		session?.timeout_override ?? status?.command_timeout_default ?? null;

	function formatBytes(bytes: number | null | undefined): string {
		if (bytes === null || bytes === undefined) return 'n/a';
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
		if (bytes < 1024 * 1024 * 1024)
			return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
	}

	async function handleToggle(event: CustomEvent<boolean>) {
		if (!session || toggling) return;
		const wanted = event.detail;
		toggling = true;
		error = null;
		try {
			const resp = await setNetwork(session.session_id, wanted);
			dispatch('networkChanged', {
				sessionId: session.session_id,
				enabled: resp.network_enabled
			});
			if (!resp.network_enabled) provisionResult = null;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Network toggle failed';
		} finally {
			toggling = false;
		}
	}

	async function handleProvision() {
		if (!session || provisioning || !requirementsPath.trim()) return;
		provisioning = true;
		error = null;
		provisionResult = null;
		try {
			provisionResult = await provisionWorkspace(session.session_id, {
				requirements_path: requirementsPath.trim(),
				venv_dir: venvDir.trim() || '.venv'
			});
		} catch (e) {
			error = e instanceof Error ? e.message : 'Provision failed';
		} finally {
			provisioning = false;
		}
	}
</script>

<div class="settings-strip">
	{#if !session}
		<p class="strip-note" role="note">
			Select a workspace to view its settings.
		</p>
	{:else}
		<div class="strip-readonly">
			<span class="strip-item">
				<Icon name="timer" size={14} />
				timeout {effectiveTimeout !== null ? `${effectiveTimeout} s` : 'default'}
				{#if session.timeout_override !== null}(per-workspace){/if}
			</span>
			<span class="strip-item">
				<Icon name="cpu" size={14} />
				caps: mem {formatBytes(status?.limit_memory_bytes)},
				{status?.limit_nproc ?? 'n/a'} procs,
				cpu {status?.limit_cpu_seconds ?? 'n/a'} s,
				disk {formatBytes(status?.disk_soft_limit_bytes)}
			</span>
		</div>

		<div class="strip-network">
			<Switch
				checked={networkOn}
				label="Workspace network (Daily-only)"
				description="Off by default. Scoped egress only -- the provision run installs a hash-pinned set; task code never touches the network."
				disabled={disabled || toggling || !networkAllowed}
				on:change={handleToggle}
			/>
			{#if !networkAllowed}
				<p class="strip-bulbe-note" role="note">
					<Icon name="shield" size={14} />
					Sandbox network is disabled in Bulbe mode: it is refused at the
					binding-layer gate (an unset or unknown mode is treated as
					Bulbe, fail-secure). Switch to Daily mode to enable it.
				</p>
			{/if}
			{#if networkOn}
				<p class="strip-warning" role="alert">
					<Icon name="alert-triangle" size={14} />
					Network is ON for this workspace. This opens a second exit that
					the copy-out approval gate does NOT cover: anything the sandbox
					can read can leave over the network without review.
					{#if hasClonedBaseline}
						Host files were cloned into this workspace -- that cloned-in
						data can leave outside the approval gate. Turn the network
						off when the provision step is done.
					{/if}
				</p>
			{/if}
		</div>

		{#if networkOn}
			<div class="strip-provision">
				<p class="strip-provision-title">
					<Icon name="package" size={14} />
					Provision dependencies (the one scoped egress)
				</p>
				<Input
					label="Requirements file (workspace-relative)"
					placeholder="requirements.txt"
					bind:value={requirementsPath}
					disabled={provisioning}
				/>
				<Input
					label="Venv directory (workspace-relative)"
					placeholder=".venv"
					bind:value={venvDir}
					disabled={provisioning}
				/>
				<Button
					variant="secondary"
					iconLeft="download"
					loading={provisioning}
					disabled={provisioning || !requirementsPath.trim()}
					ariaLabel="Run the provision install"
					on:click={handleProvision}
				>
					Run provision install
				</Button>
				<p class="strip-note" role="note">
					Every line must be an exact name==version pin carrying
					--hash=sha256: hashes; option lines are refused and nothing
					installs on a partial validation. Installs run with
					--require-hashes --only-binary=:all: (no build hooks).
				</p>
				{#if provisionResult}
					{#if provisionResult.blocked}
						<p class="strip-result strip-result-blocked" role="status">
							Provision refused: {provisionResult.block_reason}
						</p>
					{:else}
						<p
							class="strip-result"
							class:strip-result-ok={provisionResult.return_code === 0}
							class:strip-result-fail={provisionResult.return_code !== 0}
							role="status"
						>
							Provision finished with rc={provisionResult.return_code}
							({provisionResult.accepted_requirements.length} pinned
							package(s){provisionResult.timed_out ? ', timed out' : ''}).
						</p>
						{#if provisionResult.stderr_tail}
							<pre class="strip-output">{provisionResult.stderr_tail}</pre>
						{/if}
					{/if}
				{/if}
			</div>
		{/if}

		{#if error}
			<InlineError message={error} />
		{/if}
	{/if}
</div>

<style>
	.settings-strip {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
	}

	.strip-readonly {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1rem;
	}

	.strip-item {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.78rem;
		color: var(--oo-fg-secondary);
	}

	.strip-network {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}

	.strip-bulbe-note {
		display: flex;
		align-items: flex-start;
		gap: 0.35rem;
		margin: 0;
		font-size: 0.78rem;
		color: var(--oo-fg-tertiary);
	}

	.strip-warning {
		display: flex;
		align-items: flex-start;
		gap: 0.35rem;
		margin: 0;
		padding: 0.5rem 0.6rem;
		font-size: 0.8rem;
		color: var(--oo-warning);
		border: 1px solid var(--oo-warning);
		border-radius: var(--oo-radius-md, 8px);
		background: var(--oo-bg-elevated);
	}

	.strip-provision {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding-top: 0.4rem;
		border-top: 1px solid var(--oo-bd-subtle);
	}

	.strip-provision-title {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		margin: 0;
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.strip-note {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
	}

	.strip-result {
		margin: 0;
		font-size: 0.78rem;
	}

	.strip-result-ok {
		color: var(--oo-success);
	}

	.strip-result-fail {
		color: var(--oo-error);
	}

	.strip-result-blocked {
		color: var(--oo-warning);
	}

	.strip-output {
		margin: 0;
		padding: 0.4rem 0.5rem;
		max-height: 8rem;
		overflow: auto;
		font-size: 0.72rem;
		background: var(--oo-bg-elevated);
		border-radius: var(--oo-radius-sm, 4px);
		color: var(--oo-fg-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
