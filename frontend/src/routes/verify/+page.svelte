<!--
  routes/verify/+page.svelte
  The single verification surface, replacing the former /claims,
  /verify-citations and /verify-answer pages. Those three were the same
  operation -- checking claim/source pairs and returning a fail-secure
  supported / unsupported / uncertain verdict -- differing only in how the pairs
  are supplied. This page offers one tool with two input modes:
    - "cited": paste a produced answer with inline [n] markers plus its ordered
      sources; the pairs are extracted automatically (former CitationVerifier).
    - "pairs": enter claim/source pairs by hand; this also covers the single-pair
      case the former ClaimVerifier handled (former AnswerVerifier).
  The mode lives in the URL (?mode=cited|pairs) so back/forward and shared links
  land on the right mode. The verifier components are reused unchanged.
-->
<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import CitationVerifier from '$lib/components/panels/CitationVerifier.svelte';
	import AnswerVerifier from '$lib/components/panels/AnswerVerifier.svelte';

	type Mode = 'cited' | 'pairs';

	// The mode is driven by the URL query so navigation and shared links are
	// stable; anything other than "pairs" falls back to the default "cited".
	$: mode = ($page.url.searchParams.get('mode') === 'pairs' ? 'pairs' : 'cited') as Mode;

	function setMode(next: Mode): void {
		if (next === mode) return;
		const url = new URL($page.url);
		url.searchParams.set('mode', next);
		goto(url, { keepFocus: true, noScroll: true });
	}
</script>

<svelte:head>
	<title>Verification</title>
</svelte:head>

<section class="verify-page">
	<header class="verify-head">
		<h1 class="verify-title">Verification</h1>
		<p class="verify-sub">
			Check claim/source pairs and get a fail-secure verdict: an ambiguous or unparseable
			result is reported as uncertain, never as supported. Choose how to supply the pairs.
		</p>
	</header>

	<div class="verify-modes" role="group" aria-label="Verification input mode">
		<button
			type="button"
			class="verify-mode-btn"
			class:active={mode === 'cited'}
			aria-pressed={mode === 'cited'}
			on:click={() => setMode('cited')}
		>
			From a cited answer
		</button>
		<button
			type="button"
			class="verify-mode-btn"
			class:active={mode === 'pairs'}
			aria-pressed={mode === 'pairs'}
			on:click={() => setMode('pairs')}
		>
			From claim/source pairs
		</button>
	</div>

	{#if mode === 'cited'}
		<CitationVerifier />
	{:else}
		<AnswerVerifier />
	{/if}
</section>

<style>
	.verify-page {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		max-width: 48rem;
		margin: 0 auto;
		padding: var(--oo-space-5);
	}

	.verify-head {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.verify-title {
		margin: 0;
		font-size: var(--oo-text-xl);
		font-weight: 700;
		color: var(--oo-fg-primary);
	}

	.verify-sub {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.verify-modes {
		display: inline-flex;
		gap: var(--oo-space-1);
		padding: var(--oo-space-1);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 0.5rem;
		align-self: flex-start;
	}

	.verify-mode-btn {
		padding: var(--oo-space-2) var(--oo-space-3);
		border: none;
		background: transparent;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		font-weight: 500;
		border-radius: 0.375rem;
		cursor: pointer;
	}

	.verify-mode-btn.active {
		background: var(--oo-bg-subtle);
		color: var(--oo-fg-primary);
	}
</style>
