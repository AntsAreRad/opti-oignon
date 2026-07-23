<!--
  CitationVerifier.svelte (the citation-verify UI half)
  The standalone surface the user drives over the route
  (POST /api/claims/verify-citations, the join of the citation extractor
  and the per-answer aggregation). The user pastes a produced answer that
  carries inline numeric citation markers [n] (1-based) plus the ordered
  sources those markers index, one source per line, and picks a model. The
  route extracts the (claim, source) pairs server-side (this component
  interprets neither the answer nor the sources), runs each pair through the
  Verification role wrapped as untrusted context, and returns the
  structured aggregate -- supported / unsupported / uncertain -- with uncertain
  the safe default. The result is shown as the aggregate verdict, then, for
  each extracted pair, the cited claim and its source alongside the per-pair
  verdict (the pairs are positionally aligned with the per-pair results).
  CONFIRMED posture: the rendered verdicts come only from the server-returned
  result, the Verify control is disabled while a request is in flight, and the
  503 availability guard surfaces as an inline error -- a verdict is never
  shown in a state the backend has not returned. The model is the user's
  selected model (the selectedModel store), with the local model list as the
  picker source. There is no mode gate (CV-D4): the surface runs identically in
  Daily and Bulbe. Design-system tokens only (--oo-*); lucide-svelte icons
  through Icon.
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
		verifyCitations,
		type ClaimVerdict,
		type CitationVerificationResult
	} from '$lib/api/citationVerification';
	import { listModels } from '$lib/api/models';
	import { selectedModel } from '$lib/stores/chatOptions';
	import { recordVerdict } from '$lib/stores/verdictHistory';
	import VerdictHistory from './VerdictHistory.svelte';

	let answer = '';
	let sourcesText = '';
	let model = '';
	let models: SelectOption[] = [];
	let verifying = false;
	let result: CitationVerificationResult | null = null;
	let localError: string | null = null;

	// The verdict presentation: a label, an icon, and a tone token class. The
	// mapping is read-only display; the verdict itself is the server's.
	const VERDICT_META: Record<ClaimVerdict, { label: string; icon: string; tone: string }> = {
		supported: { label: 'Supported', icon: 'shield-check', tone: 'ok' },
		unsupported: { label: 'Unsupported', icon: 'shield-alert', tone: 'bad' },
		uncertain: { label: 'Uncertain', icon: 'help-circle', tone: 'warn' }
	};

	// The ordered sources: one per non-empty line, in marker-index order. The
	// answer's [n] markers are 1-based indices into this sequence.
	$: sources = sourcesText
		.split('\n')
		.map((line) => line.trim())
		.filter((line) => line.length > 0);

	$: canVerify = answer.trim().length > 0 && sources.length > 0 && !verifying;

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
			result = await verifyCitations(answer, sources, model || null);
			recordVerdict({
				surface: 'citation',
				verdict: result.verdict,
				ok: result.ok,
				summary:
					String(result.results.length) +
					(result.results.length === 1 ? ' pair' : ' pairs')
			});
		} catch (err: unknown) {
			localError = err instanceof Error ? err.message : 'Verification failed';
		} finally {
			verifying = false;
		}
	}
</script>

<Card variant="raised" padding="lg" class="citation-verifier">
	<div class="cv-head">
		<Icon name="shield-check" size="sm" />
		<span class="cv-title">Verify an answer's citations</span>
	</div>

	<label class="cv-field">
		<span class="cv-label">Answer</span>
		<textarea
			class="cv-textarea"
			rows="6"
			bind:value={answer}
			placeholder="The produced answer, with inline [n] citation markers"
		></textarea>
	</label>

	<label class="cv-field">
		<span class="cv-label">Sources (one per line, in [n] order)</span>
		<textarea
			class="cv-textarea"
			rows="6"
			bind:value={sourcesText}
			placeholder="Source [1] on the first line, source [2] on the second, ..."
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
		<Card variant="flat" padding="sm">
			<div class="cv-aggregate" data-tone={VERDICT_META[result.verdict].tone}>
				<Icon name={VERDICT_META[result.verdict].icon} size="sm" />
				<span class="cv-aggregate-label">{VERDICT_META[result.verdict].label}</span>
				<span class="cv-aggregate-meta">answer verdict</span>
			</div>
			{#if !result.ok && result.reason}
				<div class="cv-notice">
					<Icon name="alert-triangle" size="sm" />
					<span>{result.reason}</span>
				</div>
			{/if}
		</Card>

		{#if result.results.length > 0}
			<div class="cv-pairs">
				{#each result.results as r, i}
					{@const pair = result.pairs[i]}
					<Card variant="flat" padding="sm">
						<div class="cv-pair">
							<div class="cv-pair-verdict" data-tone={VERDICT_META[r.verdict].tone}>
								<Icon name={VERDICT_META[r.verdict].icon} size="sm" />
								<span class="cv-pair-label">{VERDICT_META[r.verdict].label}</span>
							</div>
							<div class="cv-pair-claim">
								<span class="cv-pair-tag">Cited claim</span>
								<p class="cv-pair-text">{pair ? pair.claim : ''}</p>
							</div>
							<div class="cv-pair-source">
								<span class="cv-pair-tag">Source</span>
								<p class="cv-pair-text">{pair ? pair.source : ''}</p>
							</div>
							{#if r.ok}
								<div class="cv-pair-raw">
									<span class="cv-pair-tag">Model reasoning</span>
									<pre class="cv-pair-rawtext">{r.raw_text}</pre>
								</div>
							{:else if r.reason}
								<div class="cv-pair-reason">{r.reason}</div>
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
			description="Paste an answer and its ordered sources, then run Verify."
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

	.cv-aggregate {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-weight: 600;
	}

	.cv-aggregate-meta {
		font-size: var(--oo-text-sm);
		font-weight: 400;
		color: var(--oo-fg-muted);
	}

	.cv-aggregate[data-tone='ok'] {
		color: var(--oo-fg-success);
	}

	.cv-aggregate[data-tone='bad'] {
		color: var(--oo-fg-danger);
	}

	.cv-aggregate[data-tone='warn'] {
		color: var(--oo-fg-warning);
	}

	.cv-notice {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		margin-top: var(--oo-space-2);
	}

	.cv-pairs {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin-top: var(--oo-space-3);
	}

	.cv-pair {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.cv-pair-verdict {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-weight: 600;
	}

	.cv-pair-verdict[data-tone='ok'] {
		color: var(--oo-fg-success);
	}

	.cv-pair-verdict[data-tone='bad'] {
		color: var(--oo-fg-danger);
	}

	.cv-pair-verdict[data-tone='warn'] {
		color: var(--oo-fg-warning);
	}

	.cv-pair-claim,
	.cv-pair-source,
	.cv-pair-raw {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.cv-pair-tag {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	.cv-pair-text {
		margin: 0;
		color: var(--oo-fg-primary);
		white-space: pre-wrap;
		word-break: break-word;
	}

	.cv-pair-rawtext {
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		font: inherit;
		color: var(--oo-fg-primary);
		max-height: 14rem;
		overflow-y: auto;
	}

	.cv-pair-reason {
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}

	.cv-history {
		margin-top: var(--oo-space-3);
	}
</style>
