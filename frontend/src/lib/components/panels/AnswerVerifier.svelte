<!--
  AnswerVerifier.svelte (the answer-verification UI half)
  The standalone surface the user drives over the route
  (POST /api/claims/verify-answer, the per-answer aggregation). The user
  builds a batch of (claim, source) pairs -- each a cited claim and the source
  it is checked against -- and picks a model. Unlike the citation surface,
  this route performs no extraction: the pairs are submitted directly, so the
  per-pair results come back positionally aligned with the submitted pairs and
  the result carries no echoed pairs. The route runs each pair through the
  verification role wrapped as untrusted context (this component interprets
  neither the claims nor the sources) and returns the structured aggregate --
  supported / unsupported / uncertain -- with uncertain the safe default. The
  result is shown as the aggregate verdict, then, for each submitted pair, the
  cited claim and its source alongside the per-pair verdict.
  CONFIRMED posture: the rendered verdicts come only from the server-returned
  result, the Verify control is disabled while a request is in flight, and the
  503 availability guard surfaces as an inline error -- a verdict is never shown
  in a state the backend has not returned. The model is the user's selected
  model (the selectedModel store), with the local model list as the picker
  source. There is no mode gate (CV-D4): the surface runs identically in Daily
  and Bulbe. Design-system tokens only (--oo-*); lucide-svelte icons through
  Icon.
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
		verifyAnswer,
		type ClaimVerdict,
		type ClaimSourcePair,
		type AnswerVerificationResult
	} from '$lib/api/answerVerification';
	import { listModels } from '$lib/api/models';
	import { selectedModel } from '$lib/stores/chatOptions';
	import { recordVerdict } from '$lib/stores/verdictHistory';
	import VerdictHistory from './VerdictHistory.svelte';

	let pairs: ClaimSourcePair[] = [{ claim: '', source: '' }];
	let model = '';
	let models: SelectOption[] = [];
	let verifying = false;
	let result: AnswerVerificationResult | null = null;
	let submitted: ClaimSourcePair[] = [];
	let localError: string | null = null;

	// The verdict presentation: a label, an icon, and a tone token class. The
	// mapping is read-only display; the verdict itself is the server's.
	const VERDICT_META: Record<ClaimVerdict, { label: string; icon: string; tone: string }> = {
		supported: { label: 'Supported', icon: 'shield-check', tone: 'ok' },
		unsupported: { label: 'Unsupported', icon: 'shield-alert', tone: 'bad' },
		uncertain: { label: 'Uncertain', icon: 'help-circle', tone: 'warn' }
	};

	// The pairs the route will verify: only those with a non-empty claim and a
	// non-empty source. The aggregate is fail-secure on an empty batch.
	$: validPairs = pairs.filter(
		(p) => p.claim.trim().length > 0 && p.source.trim().length > 0
	);

	$: canVerify = validPairs.length > 0 && !verifying;

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

	function addPair(): void {
		pairs = [...pairs, { claim: '', source: '' }];
	}

	function removePair(index: number): void {
		pairs = pairs.filter((_, i) => i !== index);
		if (pairs.length === 0) pairs = [{ claim: '', source: '' }];
	}

	async function verify(): Promise<void> {
		if (!canVerify) return;
		verifying = true;
		result = null;
		localError = null;
		// Capture the submitted batch so the per-pair results render aligned.
		submitted = validPairs.map((p) => ({ claim: p.claim, source: p.source }));
		try {
			result = await verifyAnswer(submitted, model || null);
			recordVerdict({
				surface: 'answer',
				verdict: result.verdict,
				ok: result.ok,
				summary: String(submitted.length) + (submitted.length === 1 ? ' pair' : ' pairs')
			});
		} catch (err: unknown) {
			localError = err instanceof Error ? err.message : 'Verification failed';
		} finally {
			verifying = false;
		}
	}
</script>

<Card variant="raised" padding="lg" class="answer-verifier">
	<div class="av-head">
		<Icon name="shield-check" size="sm" />
		<span class="av-title">Verify an answer's cited claims</span>
	</div>

	<div class="av-pairs-input">
		{#each pairs as pair, i}
			<Card variant="flat" padding="sm">
				<div class="av-pair-input">
					<div class="av-pair-input-head">
						<span class="av-pair-input-index">Pair {i + 1}</span>
						<button
							type="button"
							class="av-pair-remove"
							on:click={() => removePair(i)}
							aria-label="Remove pair"
						>
							<Icon name="x" size="sm" />
						</button>
					</div>
					<label class="av-field">
						<span class="av-label">Cited claim</span>
						<textarea
							class="av-textarea"
							rows="2"
							bind:value={pair.claim}
							placeholder="The cited claim to check"
						></textarea>
					</label>
					<label class="av-field">
						<span class="av-label">Source</span>
						<textarea
							class="av-textarea"
							rows="3"
							bind:value={pair.source}
							placeholder="The source the claim is checked against"
						></textarea>
					</label>
				</div>
			</Card>
		{/each}
	</div>

	<div class="av-add">
		<Button variant="secondary" size="sm" on:click={addPair}>
			<Icon name="plus" size="sm" />
			Add pair
		</Button>
	</div>

	<div class="av-controls">
		<div class="av-model">
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
		<Card variant="flat" padding="sm">
			<div class="av-aggregate" data-tone={VERDICT_META[result.verdict].tone}>
				<Icon name={VERDICT_META[result.verdict].icon} size="sm" />
				<span class="av-aggregate-label">{VERDICT_META[result.verdict].label}</span>
				<span class="av-aggregate-meta">answer verdict</span>
			</div>
			{#if !result.ok && result.reason}
				<div class="av-notice">
					<Icon name="alert-triangle" size="sm" />
					<span>{result.reason}</span>
				</div>
			{/if}
		</Card>

		{#if result.results.length > 0}
			<div class="av-results">
				{#each result.results as r, i}
					{@const pair = submitted[i]}
					<Card variant="flat" padding="sm">
						<div class="av-result">
							<div class="av-result-verdict" data-tone={VERDICT_META[r.verdict].tone}>
								<Icon name={VERDICT_META[r.verdict].icon} size="sm" />
								<span class="av-result-label">{VERDICT_META[r.verdict].label}</span>
							</div>
							<div class="av-result-claim">
								<span class="av-result-tag">Cited claim</span>
								<p class="av-result-text">{pair ? pair.claim : ''}</p>
							</div>
							<div class="av-result-source">
								<span class="av-result-tag">Source</span>
								<p class="av-result-text">{pair ? pair.source : ''}</p>
							</div>
							{#if r.ok}
								<div class="av-result-raw">
									<span class="av-result-tag">Model reasoning</span>
									<pre class="av-result-rawtext">{r.raw_text}</pre>
								</div>
							{:else if r.reason}
								<div class="av-result-reason">{r.reason}</div>
							{/if}
						</div>
					</Card>
				{/each}
			</div>
		{/if}
	{:else if !verifying}
		<EmptyState
			icon="shield-check"
			title="No verdict yet"
			description="Add one or more claim and source pairs, then run Verify."
		/>
	{/if}

	<div class="av-history">
		<VerdictHistory />
	</div>
</Card>

<style>
	.av-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-tertiary);
		margin-bottom: var(--oo-space-3);
	}

	.av-title {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.av-pairs-input {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin-bottom: var(--oo-space-2);
	}

	.av-pair-input {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.av-pair-input-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.av-pair-input-index {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-secondary);
	}

	.av-pair-remove {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: var(--oo-space-1);
		color: var(--oo-fg-muted);
		background: transparent;
		border: none;
		border-radius: var(--oo-radius-sm);
		cursor: pointer;
	}

	.av-pair-remove:hover {
		color: var(--oo-fg-danger);
	}

	.av-field {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.av-label {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-secondary);
	}

	.av-textarea {
		width: 100%;
		resize: vertical;
		font: inherit;
		color: var(--oo-fg-primary);
		background-color: var(--oo-bg-input);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		padding: var(--oo-space-2);
	}

	.av-textarea:focus {
		outline: none;
		border-color: var(--oo-bd-strong);
	}

	.av-add {
		margin-bottom: var(--oo-space-3);
	}

	.av-controls {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-3);
	}

	.av-model {
		flex: 1 1 auto;
		max-width: 20rem;
	}

	.av-aggregate {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-weight: 600;
	}

	.av-aggregate-meta {
		font-size: var(--oo-text-sm);
		font-weight: 400;
		color: var(--oo-fg-muted);
	}

	.av-aggregate[data-tone='ok'] {
		color: var(--oo-fg-success);
	}

	.av-aggregate[data-tone='bad'] {
		color: var(--oo-fg-danger);
	}

	.av-aggregate[data-tone='warn'] {
		color: var(--oo-fg-warning);
	}

	.av-notice {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		margin-top: var(--oo-space-2);
	}

	.av-results {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin-top: var(--oo-space-3);
	}

	.av-result {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.av-result-verdict {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-weight: 600;
	}

	.av-result-verdict[data-tone='ok'] {
		color: var(--oo-fg-success);
	}

	.av-result-verdict[data-tone='bad'] {
		color: var(--oo-fg-danger);
	}

	.av-result-verdict[data-tone='warn'] {
		color: var(--oo-fg-warning);
	}

	.av-result-claim,
	.av-result-source,
	.av-result-raw {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.av-result-tag {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	.av-result-text {
		margin: 0;
		color: var(--oo-fg-primary);
		white-space: pre-wrap;
		word-break: break-word;
	}

	.av-result-rawtext {
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		font: inherit;
		color: var(--oo-fg-primary);
		max-height: 14rem;
		overflow-y: auto;
	}

	.av-result-reason {
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}

	.av-history {
		margin-top: var(--oo-space-3);
	}
</style>
