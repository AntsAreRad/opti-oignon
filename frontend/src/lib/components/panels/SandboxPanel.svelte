<!--
  SandboxPanel.svelte (Sandbox Workspace cycle, Bloc 1)
  The workspace manager panel (spec sections 4.2 and 10), built on the lib/ds
  primitives, mounted in the chat-side right panel next to AgentPanel
  (FRD-03's runtime home). It surfaces the sandbox system status (backend,
  bwrap, sessions x/max), a create form (optional label, per-sandbox timeout),
  and the workspace list (SandboxWorkspaceList) with stop / delete / select.
  Select binds the workspace to the ACTIVE conversation over the binding
  routes: an explicit user action -- a conversation has no workspace until one
  is created or attached, at most one active conversation per workspace, and
  the agent run then reaches the sandboxed tools through that binding
  (ATL-02). The network field is live since Bloc 4: the settings
  strip owns the Daily-only, user-activated toggle. Updates
  announce through an aria-live region. Design-system tokens only (--oo-*);
  lucide icons through Icon. Registered in FRONTEND_REDESIGN_SPEC.md.
  (Bloc 3): hosts the diff review + apply card (SandboxDiffReview) on
  the same explicit target Select; review and apply are user actions only.
  (Bloc 4): hosts the per-workspace settings strip
  (SandboxSettingsStrip) on the same explicit target Select -- timeout and
  caps read-only, the network toggle disabled under Bulbe with the refusal
  stated honestly, the exfiltration warning when on, and the provision row.
-->
<script lang="ts">
	import { Button, Card, Icon, Input, InlineError, Select } from '$lib/ds';
	import SandboxWorkspaceList from './SandboxWorkspaceList.svelte';
	import SandboxUploadZone from './SandboxUploadZone.svelte';
	import SandboxHostExplorer from './SandboxHostExplorer.svelte';
	import SandboxDiffReview from './SandboxDiffReview.svelte';
	import SandboxSettingsStrip from './SandboxSettingsStrip.svelte';
	import {
		getSandboxStatus,
		createSandbox,
		listSessions,
		destroySandbox,
		stopSandboxCommand,
		bindConversation,
		unbindConversation
	} from '$lib/api/sandbox';
	import type { SandboxSessionInfo, SandboxStatusResponse } from '$lib/types';
	import { activeConversationId } from '$lib/stores/conversations';
	import { workspaceBinding } from '$lib/stores/workspaceBinding';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import { handleApiError, parseApiError } from '$lib/api/errorHandler';

	let status: SandboxStatusResponse | null = null;
	let sessions: SandboxSessionInfo[] = [];
	let loading = false;
	let error: string | null = null;
	let liveMessage = '';

	let newLabel = '';
	let newTimeout = '';
	let creating = false;
	let busyId: string | null = null;

	// (Bloc 2): the copy-in target. Defaults to the workspace bound to
	// the active conversation, else the first workspace; the Select makes
	// the choice explicit. Upload and clone are user actions only.
	let copyTargetId = '';

	$: convId = $activeConversationId ? String($activeConversationId) : null;
	$: atCapacity =
		status !== null && status.active_sessions >= status.max_sessions;
	$: copyTargetOptions = sessions.map((s) => ({
		value: s.session_id,
		label: s.label ? `${s.label} (${s.session_id})` : s.session_id
	}));
	$: if (copyTargetId && !sessions.some((s) => s.session_id === copyTargetId)) {
		copyTargetId = '';
	}
	$: if (!copyTargetId && sessions.length > 0) {
		const bound = sessions.find(
			(s) => convId !== null && s.bound_conversation_id === convId
		);
		copyTargetId = (bound ?? sessions[0]).session_id;
	}

	async function load() {
		loading = true;
		error = null;
		try {
			const [st, list] = await Promise.all([getSandboxStatus(), listSessions()]);
			status = st;
			sessions = list;
		} catch (e) {
			error = parseApiError(e, 'loading the sandbox state').message;
		} finally {
			loading = false;
		}
	}

	async function handleCreate() {
		creating = true;
		try {
			const timeoutValue = newTimeout.trim() === '' ? null : Number(newTimeout);
			const created = await createSandbox({
				label: newLabel.trim(),
				timeout: timeoutValue !== null && Number.isFinite(timeoutValue) ? timeoutValue : null
			});
			liveMessage = `Workspace ${created.session_id} created`;
			toastSuccess(liveMessage);
			newLabel = '';
			newTimeout = '';
			await load();
		} catch (e) {
			handleApiError(e, 'creating the workspace');
		} finally {
			creating = false;
		}
	}

	async function handleStop(event: CustomEvent<{ sessionId: string }>) {
		const sessionId = event.detail.sessionId;
		busyId = sessionId;
		try {
			const result = await stopSandboxCommand(sessionId);
			liveMessage = result.stopped
				? `Running command stopped in ${sessionId}; the workspace persists`
				: `Nothing was running in ${sessionId}`;
			toastSuccess(liveMessage);
			await load();
		} catch (e) {
			handleApiError(e, 'stopping the command');
		} finally {
			busyId = null;
		}
	}

	async function handleDestroy(event: CustomEvent<{ sessionId: string }>) {
		const sessionId = event.detail.sessionId;
		busyId = sessionId;
		try {
			await destroySandbox(sessionId);
			liveMessage = `Workspace ${sessionId} destroyed`;
			toastSuccess(liveMessage);
			await load();
		} catch (e) {
			handleApiError(e, 'destroying the workspace');
		} finally {
			busyId = null;
		}
	}

	async function handleSelect(event: CustomEvent<{ sessionId: string }>) {
		const sessionId = event.detail.sessionId;
		if (!convId) {
			toastError('Open a conversation before binding a workspace');
			return;
		}
		busyId = sessionId;
		try {
			await bindConversation({ conversation_id: convId, session_id: sessionId });
			liveMessage = `Workspace ${sessionId} bound to this conversation`;
			toastSuccess(liveMessage);
			workspaceBinding.applyBound(convId, sessionId);
			await load();
		} catch (e) {
			// Surface the server detail (409: held by another conversation)
			// instead of the bare status line, and resync the stale list
			// that invited the doomed action.
			const parsed = handleApiError(e, 'binding the workspace');
			if (parsed.status === 409) {
				await load();
			}
		} finally {
			busyId = null;
		}
	}

	async function handleUnbind(
		event: CustomEvent<{ sessionId: string; conversationId: string }>
	) {
		const { sessionId, conversationId } = event.detail;
		busyId = sessionId;
		try {
			await unbindConversation(conversationId);
			liveMessage = `Workspace ${sessionId} unbound`;
			toastSuccess(liveMessage);
			workspaceBinding.applyUnbound(conversationId);
			await load();
		} catch (e) {
			handleApiError(e, 'unbinding the workspace');
		} finally {
			busyId = null;
		}
	}

	function handleUploaded(
		event: CustomEvent<{ sessionId: string; count: number; bytes: number }>
	) {
		const { sessionId, count } = event.detail;
		liveMessage = `${count} file(s) uploaded into ${sessionId}`;
		toastSuccess(liveMessage);
		void load();
	}

	function handleCloned(
		event: CustomEvent<{ sessionId: string; dest: string; files: number }>
	) {
		const { sessionId, dest, files } = event.detail;
		liveMessage = `Cloned ${files} file(s) into ${sessionId}/${dest}`;
		toastSuccess(liveMessage);
		void load();
	}

	function handleApplied(
		event: CustomEvent<{ sessionId: string; applied: number; deleted: number }>
	) {
		const { sessionId, applied, deleted } = event.detail;
		liveMessage = `Applied ${applied} write(s) and ${deleted} deletion(s) from ${sessionId} to the host`;
		toastSuccess(liveMessage);
		void load();
	}

	// (Bloc 4): the settings strip targets the same explicit Select.
	$: copyTargetSession =
		sessions.find((s) => s.session_id === copyTargetId) ?? null;

	function handleNetworkChanged(
		event: CustomEvent<{ sessionId: string; enabled: boolean }>
	) {
		const { sessionId, enabled } = event.detail;
		liveMessage = `Workspace ${sessionId} network ${enabled ? 'enabled' : 'disabled'}`;
		if (enabled) {
			toastSuccess(liveMessage);
		}
		void load();
	}

	// The panel lives in the chat layout and survives navigation: the
	// workspace list must follow the active conversation instead of
	// freezing on its mount-time state (binding badges, Select/Unbind
	// availability and the copy-in target all derive from it).
	let loadedForConv: string | null | undefined;
	$: if (convId !== loadedForConv) {
		loadedForConv = convId;
		void load();
	}
</script>

<section class="sandbox-panel" aria-label="Sandbox workspaces">
	<header class="sandbox-header">
		<div class="sandbox-title">
			<Icon name="box" />
			<h2>Workspaces</h2>
		</div>
		<Button variant="ghost" on:click={load} disabled={loading}>
			<Icon name="refresh-cw" />
			Refresh
		</Button>
	</header>

	<p class="sandbox-live" aria-live="polite">{liveMessage}</p>

	{#if error}
		<InlineError message={error} />
	{/if}

	{#if status}
		<div class="sandbox-status" role="status">
			<span title="Isolation backend">backend {status.isolation_backend}</span>
			<span
				class:status-ok={status.bwrap_available}
				class:status-warn={!status.bwrap_available}
				title="bubblewrap availability; without it strict mode refuses execution"
			>
				bwrap {status.bwrap_available ? 'available' : 'unavailable'}
			</span>
			<span title="Active sessions against the concurrency cap">
				sessions {status.active_sessions}/{status.max_sessions}
			</span>
		</div>
	{/if}

	<Card>
		<div class="sandbox-create">
			<Input
				label="Label"
				placeholder="Optional workspace label"
				bind:value={newLabel}
				disabled={creating}
			/>
			<Input
				label="Timeout (s)"
				placeholder="Per-sandbox command timeout"
				bind:value={newTimeout}
				disabled={creating}
			/>
			<Button
				variant="primary"
				iconLeft="plus"
				loading={creating}
				disabled={creating || atCapacity}
				ariaLabel="Create a new workspace"
				on:click={handleCreate}
			>
				Create workspace
			</Button>
			{#if atCapacity}
				<p class="sandbox-cap-note" role="note">
					Concurrency cap reached; stop or delete a workspace to free a slot.
				</p>
			{/if}
		</div>
	</Card>

	<SandboxWorkspaceList
		{sessions}
		activeConversationId={convId}
		{busyId}
		on:stop={handleStop}
		on:destroy={handleDestroy}
		on:select={handleSelect}
		on:unbind={handleUnbind}
	/>

	<Card>
		<div class="sandbox-copyin">
			<h3 class="copyin-title">
				<Icon name="download" />
				Copy in
			</h3>
			{#if sessions.length === 0}
				<p class="sandbox-note" role="note">
					Create a workspace to copy files into it.
				</p>
			{:else}
				<Select
					label="Target workspace"
					options={copyTargetOptions}
					bind:value={copyTargetId}
				/>
				<SandboxUploadZone
					sessionId={copyTargetId || null}
					on:uploaded={handleUploaded}
				/>
				<SandboxHostExplorer
					sessionId={copyTargetId || null}
					on:cloned={handleCloned}
				/>
			{/if}
		</div>
	</Card>

	<Card>
		<div class="sandbox-review">
			<h3 class="copyin-title">
				<Icon name="file-diff" />
				Review and apply
			</h3>
			{#if sessions.length === 0}
				<p class="sandbox-note" role="note">
					Create a workspace to review its changes.
				</p>
			{:else}
				<SandboxDiffReview
					sessionId={copyTargetId || null}
					on:applied={handleApplied}
				/>
			{/if}
		</div>
	</Card>

	<Card>
		<div class="sandbox-settings">
			<h3 class="copyin-title">
				<Icon name="settings" />
				Workspace settings
			</h3>
			{#if sessions.length === 0}
				<p class="sandbox-note" role="note">
					Create a workspace to view its settings.
				</p>
			{:else}
				<SandboxSettingsStrip
					session={copyTargetSession}
					{status}
					on:networkChanged={handleNetworkChanged}
				/>
			{/if}
		</div>
	</Card>

	<p class="sandbox-note" role="note">
		Binding a workspace to the active conversation routes the agent's
		filesystem, shell and code tools through that isolated sandbox. Network
		is off by default and Daily-only: enabling it is your explicit,
		per-workspace choice (the settings strip), scoped to the provision
		install -- and it opens a second exit the copy-out approval gate does
		not cover. Results leave toward the host only behind your approval.
	</p>
</section>

<style>
	.sandbox-panel {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		padding: 1rem;
		height: 100%;
		overflow-y: auto;
	}
	.sandbox-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	.sandbox-title {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: var(--oo-fg-primary);
	}
	.sandbox-title h2 {
		margin: 0;
		font-size: 1rem;
		font-weight: 600;
	}
	.sandbox-live {
		position: absolute;
		width: 1px;
		height: 1px;
		margin: -1px;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
	}
	.sandbox-status {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
		font-size: 0.75rem;
		color: var(--oo-fg-secondary);
	}
	.status-ok {
		color: var(--oo-success);
	}
	.status-warn {
		color: var(--oo-warning);
	}
	.sandbox-create {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.sandbox-copyin,
	.sandbox-review {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.sandbox-settings {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.copyin-title {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0;
		font-size: 0.85rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
	}
	.sandbox-cap-note,
	.sandbox-note {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
	}
</style>
