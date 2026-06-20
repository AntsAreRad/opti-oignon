<!--
  MessageSkeleton.svelte
  Animated placeholder for messages while they load. Renders a few
  skeleton blocks with varied widths.
-->
<script lang="ts">
	export let count: number = 3;
</script>

<div class="space-y-4" role="status" aria-label="Loading messages">
	{#each Array(count) as _, i}
		<div class="flex {i % 2 === 0 ? 'justify-start' : 'justify-end'}">
			<div class="max-w-[85%] rounded-xl px-4 py-3 {i % 2 === 0 ? 'bg-surface-800' : 'bg-accent-600/10'}">
				{#if i % 2 === 0}
					<!-- Skeleton: model name -->
					<div class="h-3 w-20 rounded bg-surface-700 animate-skeleton mb-2" />
				{/if}
				<!-- Skeleton: text lines -->
				<div class="space-y-2">
					<div class="h-3.5 rounded bg-surface-700 animate-skeleton"
						style="width: {i % 2 === 0 ? '260px' : '180px'}; max-width: 100%;" />
					<div class="h-3.5 rounded bg-surface-700 animate-skeleton"
						style="width: {i % 2 === 0 ? '200px' : '140px'}; max-width: 100%;" />
					{#if i % 2 === 0}
						<div class="h-3.5 w-24 rounded bg-surface-700 animate-skeleton" style="max-width: 100%;" />
					{/if}
				</div>
			</div>
		</div>
	{/each}
	<span class="sr-only">Loading messages...</span>
</div>
