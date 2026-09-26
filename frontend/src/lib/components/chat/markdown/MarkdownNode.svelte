<script lang="ts">
	/**
	 * MarkdownNode.svelte -- one node of a reply's tree (lib/markdown/tree.ts),
	 * and its children through svelte:self. Every kind of the closed set has its
	 * branch; text reaches the page only through interpolation, which escapes
	 * every character. A link carries the destination the tree accepted (an
	 * absolute http, https or mailto URL) and opens in a new tab with no
	 * referrer. A raw paragraph (source shown as written) keeps its line breaks
	 * and indentation. A code block and a table have components of their own.
	 * While the reply streams, the node that holds its last character is handed
	 * the caret, and each node hands it on to its last child: it is drawn after
	 * the last text, inside the last block.
	 */
	import type { MdNode } from '$lib/markdown/tree';
	import CodeBlock from './CodeBlock.svelte';
	import MarkdownTable from './MarkdownTable.svelte';
	import Caret from './Caret.svelte';

	export let node: MdNode;
	/** True when this node holds the reply's last character while it streams. */
	export let caret = false;

	$: children = 'children' in node ? (node.children as MdNode[]) : [];
	$: last = children.length - 1;
	$: empty = caret && children.length === 0;
</script>

{#if node.kind === 'text'}{node.text}{#if caret}<Caret />{/if}{:else if node.kind === 'br'}<br
	/>{#if caret}<Caret />{/if}{:else if node.kind === 'strong'}<strong
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
			/>{/if}</strong
	>{:else if node.kind === 'em'}<em
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
			/>{/if}</em
	>{:else if node.kind === 'del'}<del
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
			/>{/if}</del
	>{:else if node.kind === 'code_inline'}<code class="oo-md-code-inline">{node.text}</code
	>{#if caret}<Caret />{/if}{:else if node.kind === 'link'}<a href={node.href} title={node.title || undefined} target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer" class="oo-md-link"
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}</a
	>{:else if node.kind === 'p'}<p class="oo-md-p" class:oo-md-raw={node.raw}
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
			/>{/if}</p
	>{:else if node.kind === 'heading'}<svelte:element this={`h${node.level}`} class="oo-md-heading"
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
			/>{/if}</svelte:element
	>{:else if node.kind === 'code_block'}<CodeBlock {node} {caret} />{:else if node.kind === 'blockquote'}<blockquote
		class="oo-md-quote"
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
			/>{/if}</blockquote
	>{:else if node.kind === 'ol'}<ol class="oo-md-list" start={node.start}
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}</ol
	>{:else if node.kind === 'ul'}<ul class="oo-md-list"
		>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}</ul
	>{:else if node.kind === 'li'}{#if node.task}<li class="oo-md-item oo-md-task"
			><span class="sr-only">{node.checked ? 'done, ' : 'not done, '}</span><svg
				class="oo-md-box"
				viewBox="0 0 16 16"
				width="16"
				height="16"
				aria-hidden="true"
				focusable="false"
				><path
					d="M3 2.5h10a.5.5 0 0 1 .5.5v10a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5V3a.5.5 0 0 1 .5-.5z"
				/>{#if node.checked}<path d="M5 8.2l2.1 2.1L11 5.8" />{/if}</svg
			><div class="oo-md-task-body">
				{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
					/>{/if}
			</div></li
		>{:else}<li class="oo-md-item"
			>{#each children as child, index}<svelte:self node={child} caret={caret && index === last} />{/each}{#if empty}<Caret
				/>{/if}</li
		>{/if}{:else if node.kind === 'table'}<MarkdownTable {node} />{#if caret}<Caret
		/>{/if}{:else if node.kind === 'hr'}<hr class="oo-md-rule" />{#if caret}<Caret />{/if}{/if}

<style>
	.oo-md-p {
		margin: 0 0 var(--oo-space-3);
	}

	.oo-md-raw {
		white-space: pre-wrap;
	}

	.oo-md-heading {
		margin: var(--oo-space-4) 0 var(--oo-space-2);
		font-weight: 600;
		line-height: 1.3;
	}

	h3.oo-md-heading {
		font-size: 1.25em;
	}

	h4.oo-md-heading {
		font-size: 1.125em;
	}

	h5.oo-md-heading,
	h6.oo-md-heading {
		font-size: 1em;
	}

	.oo-md-code-inline {
		padding: 0.1em 0.35em;
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-bg-code);
		font-family: var(--oo-font-mono);
		font-size: max(12px, 0.9em);
	}

	.oo-md-link {
		color: var(--oo-acc-400);
		text-decoration: underline;
		text-underline-offset: 2px;
	}

	.oo-md-quote {
		margin: 0 0 var(--oo-space-3);
		padding: var(--oo-space-2) var(--oo-space-4);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-code);
		color: var(--oo-fg-secondary);
	}

	.oo-md-list {
		margin: 0 0 var(--oo-space-3);
		padding-left: 1.5em;
	}

	ul.oo-md-list {
		list-style: disc;
	}

	ol.oo-md-list {
		list-style: decimal;
	}

	.oo-md-item {
		margin: var(--oo-space-1) 0;
	}

	.oo-md-task {
		display: flex;
		align-items: flex-start;
		gap: 0.5em;
		list-style: none;
		margin-left: -1.5em;
	}

	.oo-md-task-body {
		flex: 1 1 auto;
		min-width: 0;
	}

	.oo-md-box {
		flex: none;
		width: 1em;
		height: 1em;
		margin-top: 0.3em;
		fill: none;
		stroke: currentColor;
		stroke-width: 1.5;
		stroke-linecap: round;
		stroke-linejoin: round;
	}

	.oo-md-rule {
		width: 40px;
		height: 3px;
		margin: var(--oo-space-4) auto;
		border: 0;
		border-radius: 2px;
		background-color: var(--oo-fg-faint);
	}
</style>
