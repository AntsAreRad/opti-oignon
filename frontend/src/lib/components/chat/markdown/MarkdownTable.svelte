<script lang="ts">
	/**
	 * MarkdownTable.svelte -- one table of a reply, in a region that scrolls
	 * sideways, takes the focus (so the keyboard can scroll it) and is named by
	 * its column count. Header cells are column headers; every cell carries its
	 * column's alignment. The header row is set apart by tone, not by a line.
	 */
	import type { TableNode } from '$lib/markdown/tree';
	import MarkdownNode from './MarkdownNode.svelte';

	export let node: TableNode;

	$: columns = node.header.length;
	$: label = `Table, ${columns} ${columns === 1 ? 'column' : 'columns'}`;
</script>

<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
<div class="oo-md-table-region" role="region" aria-label={label} tabindex="0">
	<table class="oo-md-table">
		<thead>
			<tr>
				{#each node.header as cell, index}
					<th scope="col" data-align={node.align[index]}
						>{#each cell as child}<MarkdownNode node={child} />{/each}</th
					>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each node.rows as row}
				<tr>
					{#each row as cell, index}
						<td data-align={node.align[index]}
							>{#each cell as child}<MarkdownNode node={child} />{/each}</td
						>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.oo-md-table-region {
		margin: 0 0 var(--oo-space-3);
		overflow-x: auto;
		border-radius: var(--oo-radius-lg);
		-webkit-overflow-scrolling: touch;
	}

	.oo-md-table {
		border-collapse: collapse;
		font-size: max(12px, 0.95em);
	}

	th,
	td {
		padding: var(--oo-space-2) var(--oo-space-3);
		text-align: left;
		vertical-align: top;
	}

	th {
		background-color: var(--oo-bg-code);
		font-weight: 600;
	}

	[data-align='left'] {
		text-align: left;
	}

	[data-align='center'] {
		text-align: center;
	}

	[data-align='right'] {
		text-align: right;
	}
</style>
