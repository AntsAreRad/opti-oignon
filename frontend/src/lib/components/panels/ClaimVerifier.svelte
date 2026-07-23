<!--
  ClaimVerifier.svelte (the claim-vs-source verification role's UI half)
  The standalone surface the user drives over the route
  (POST /api/claims/verify, the verification role). The user pastes a
  model-generated claim and its cited source and picks a model; the role wraps
  both as untrusted context server-side (this component interprets neither) and
  returns a fail-secure verdict -- supported / unsupported / uncertain -- with
  uncertain the safe default on an ambiguous reply. The structured result is
  shown alongside the cited source: the mapped verdict, the model's raw text on
  success, or the reason on a clean fail-secure failure. CONFIRMED posture: the
  rendered verdict comes only from the server-returned result, the Verify
  control is disabled while a request is in flight, and the 503 availability
  guard surfaces as an inline error -- a verdict is never shown in a state the
  backend has not returned. The model is the user's selected model (the
  selectedModel store), with the local model list as the picker source. There
  is no mode gate (CV-D4): the role runs identically in Daily and Bulbe.
  Design-system tokens only (--oo-*); lucide-svelte icons through Icon.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { get } from 'svelte/store';
	import {
		Button,
		Card,
		Icon,
		EmptyState,
		InlineError,
		Select,
		type SelectOption
	} from '$lib/ds';
	import {
		verifyClaim,
		type ClaimVerdict,
		type ClaimVerificationResult
	} from '$lib/api/claimVerification';
	import { listModels } from '$lib/api/models';
	import { selectedModel } from '$lib/stores/chatOptions';
	import { recordVerdict } from '$lib/stores/verdictHistory';
	import VerdictHistory from './VerdictHistory.svelte';

	let claim = '';
	let source = '';
	let model = '';
	let models: SelectOption[] = [];
	let verifying = false;
	let result: ClaimVerificationResult | null = null;
	let localError: string | null = null;

	// The verdict presentation: a label, an icon, and a tone token class. The
	// mapping is read-only display; the verdict itself is the server's.
	const VERDICT_META: Record<ClaimVerdict, { label: string; icon: string; tone: string }> = {
		supported: { label: 'Supported', icon: 'shield-check', tone: 'ok' },
		unsupported: { label: 'Unsupported', icon: 'shield-alert', tone: 'bad' },
		uncertain: { label: 'Uncertain', icon: 'help-circle', tone: 'warn' }
	};

	$: canVerify = claim.trim().length > 0 && source.trim().length > 0 && !verifying;

	onMount(async () => {
		// Seed the model from the user's selected model, then offer the local
		// model list as the picker source. Read-only: this surface edits no store.
		model = get(selectedModel) || '';
		try {
			const resp = await listModels();
			const names = (resp?.models ?? []).map((m) => m.name);
			models = names.map((name) => ({ value: name, label: name }));
			if (!model && names.length > 0) model = names[0];
		} catch {
			models = [];
		}
	});

	async function verify(): Promise<void> {
		if (!canVerify) return;
		verifying = true;
		result = null;
		localError = null;
		try {
			result = await verifyClaim(claim, source, model || null);
			recordVerdict({
				surface: 'claim',
				verdict: result.verdict,
				ok: result.ok,
				summary: '1 claim'
			});
		} catch (err: unknown) {
			localError = err instanceof Error ? err.message : 'Verification failed';
		} finally {
			verifying = false;
		}
	}
</script>

<Card variant="raised" padding="lg" class="claim-verifier">
	<div class="cv-head">
		<Icon name="shield-check" size="sm" />
		<span class="cv-title">Verify a claim against its source</span>
	</div>

	<label class="cv-field">
		<span class="cv-label">Claim</span>
		<textarea
			class="cv-textarea"
			rows="3"
			bind:value={claim}
			placeholder="The model-generated claim to check"
		></textarea>
	</label>

	<label class="cv-field">
		<span class="cv-label">Source</span>
		<textarea
			class="cv-textarea"
			rows="6"
			bind:value={source}
			placeholder="The cited source the claim must be supported by"
		></textarea>
	</label>

	<div class="cv-controls">
		<div class="cv-model">
			<Select
				label="Model"
				bind:value={model}
				options={models}
				placeholder="Select a model"
				size="sm"
			/>
		</div>
		<Button variant="primary" loading={verifying} disabled={!canVerify} on:click={verify}>
			Verify
		</Button>
	</div>

	{#if localError}
		<InlineError message={localError} />
	{:else if result}
		{#if result.ok}
			<Card variant="flat" padding="sm">
				<div class="cv-result">
					<div class="cv-verdict" data-tone={VERDICT_META[result.verdict].tone}>
						<Icon name={VERDICT_META[result.verdict].icon} size="sm" />
						<span class="cv-verdict-label">{VERDICT_META[result.verdict].label}</span>
					</div>
					<div class="cv-raw">
						<span class="cv-raw-label">Model reasoning</span>
						<pre class="cv-raw-text">{result.raw_text}</pre>
					</div>
				</div>
			</Card>
		{:else}
			<Card variant="flat" padding="sm">
				<div class="cv-notice">
					<Icon name="alert-triangle" size="sm" />
					<span>{result.reason || 'The verification could not be completed.'}</span>
				</div>
			</Card>
		{/if}
	{:else if !verifying}
		<EmptyState
			icon="shield-check"
			title="No verdict yet"
			description="Paste a claim and its source, then run Verify."
		/>
	{/if}

	<div class="cv-history">
		<VerdictHistory />
	</div>
</Card>

<style>
	.cv-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-tertiary);
		margin-bottom: var(--oo-space-3);
	}

	.cv-title {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.cv-field {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		margin-bottom: var(--oo-space-3);
	}

	.cv-label {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-secondary);
	}

	.cv-textarea {
		width: 100%;
		resize: vertical;
		font: inherit;
		color: var(--oo-fg-primary);
		background-color: var(--oo-bg-input);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		padding: var(--oo-space-2);
	}

	.cv-textarea:focus {
		outline: none;
		border-color: var(--oo-bd-strong);
	}

	.cv-controls {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-3);
	}

	.cv-model {
		flex: 1 1 auto;
		max-width: 20rem;
	}

	.cv-result {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.cv-verdict {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-weight: 600;
	}

	.cv-verdict[data-tone='ok'] {
		color: var(--oo-fg-success);
	}

	.cv-verdict[data-tone='bad'] {
		color: var(--oo-fg-danger);
	}

	.cv-verdict[data-tone='warn'] {
		color: var(--oo-fg-warning);
	}

	.cv-raw {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.cv-raw-label {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	.cv-raw-text {
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		font: inherit;
		color: var(--oo-fg-primary);
		max-height: 18rem;
		overflow-y: auto;
	}

	.cv-notice {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}

	.cv-history {
		margin-top: var(--oo-space-3);
	}
</style>
