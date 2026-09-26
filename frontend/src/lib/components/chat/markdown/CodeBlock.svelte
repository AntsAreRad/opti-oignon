<script context="module" lang="ts">
	let codeBlocks = 0;
</script>

<script lang="ts">
	/**
	 * CodeBlock.svelte -- one code block of a reply: a caption with the block's
	 * label on the left and Copy on the right, then the code, highlighted in two
	 * roles (lib/markdown/highlight.ts), scrollable and focusable. The figure is
	 * named by its label alone. Copy writes the block's code, as the tree holds
	 * it, and says "Copied" through a status region once the write resolved, or
	 * "Copy failed"; it is absent while the block is still arriving, and the
	 * streaming caret is then drawn after the code.
	 */
	import { onDestroy } from 'svelte';
	import type { CodeBlockNode } from '$lib/markdown/tree';
	import { highlight } from '$lib/markdown/highlight';
	import TextButton from '$lib/ds/TextButton.svelte';
	import Caret from './Caret.svelte';

	export let node: CodeBlockNode;
	/** True when the reply's last character is this block's, while it streams. */
	export let caret = false;

	codeBlocks += 1;
	const labelId = `oo-md-code-${codeBlocks}`;

	let status = '';
	let timer: ReturnType<typeof setTimeout> | null = null;

	$: spans = highlight(node.code, node.lang);

	async function copy(): Promise<void> {
		try {
			await navigator.clipboard.writeText(node.code);
			status = 'Copied';
		} catch {
			status = 'Copy failed';
		}
		if (timer !== null) clearTimeout(timer);
		timer = setTimeout(() => {
			status = '';
			timer = null;
		}, 2000);
	}

	onDestroy(() => {
		if (timer !== null) clearTimeout(timer);
	});
</script>

<figure class="oo-md-code" aria-labelledby={labelId}>
	<figcaption class="oo-md-code-caption">
		<span class="oo-md-code-label" id={labelId}>{node.lang || 'Code'}</span>
		<span class="oo-md-code-status" role="status">{status}</span>
		{#if node.closed}
			<TextButton on:click={copy}>Copy<span class="sr-only">{' code'}</span></TextButton>
		{/if}
	</figcaption>
	<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
	<pre class="oo-md-code-body" tabindex="0"><code
			>{#each spans as span}<span class="oo-md-syn-{span.role}">{span.text}</span>{/each}{#if caret}<Caret
				/>{/if}</code
		></pre>
</figure>

<style>
	.oo-md-code {
		margin: 0 0 var(--oo-space-3);
		border-radius: 22px;
		background-color: var(--oo-bg-code);
		overflow: hidden;
	}

	.oo-md-code-caption {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		min-height: 36px;
		padding: 0 var(--oo-space-2) 0 var(--oo-space-4);
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}

	.oo-md-code-label {
		flex: 1 1 auto;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.oo-md-code-body {
		margin: 0;
		padding: 0 var(--oo-space-4) var(--oo-space-4);
		overflow-x: auto;
		font-family: var(--oo-font-mono);
		font-size: 14px;
		line-height: 21px;
		white-space: pre;
		tab-size: 4;
		-webkit-overflow-scrolling: touch;
	}

	.oo-md-code-body:focus-visible {
		outline-offset: -2px;
	}

	.oo-md-syn-keyword {
		color: var(--oo-code-keyword);
	}

	.oo-md-syn-name {
		color: var(--oo-code-name);
	}
</style>
