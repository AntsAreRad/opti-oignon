<!--
  ProjectContextLine.svelte
  Under an answer, where its project's context came from and what the turn
  said of it, as the server kept them with the answer (`project_context`):
  in the chat loader's own words, files counted and never named; nothing
  for an answer without a project, or for anything that is not those facts.
-->
<script lang="ts">
	import { projectFactsLines, said } from '$lib/chat/loaderWords';

	export let facts: unknown = null;

	$: lines = projectFactsLines(facts).map((line) => said(line));
</script>

{#if lines.length}
	<p class="oo-project-context-line">{lines.join(' ')}</p>
{/if}

<style>
	/* The run summary's own size and colour: the answer's footer, one voice. */
	.oo-project-context-line {
		margin: var(--oo-space-2) 0 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
</style>
