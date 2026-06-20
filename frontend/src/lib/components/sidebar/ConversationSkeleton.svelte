<!--
  ConversationSkeleton.svelte
  Placeholder anime pour la liste de conversations en chargement.
  Affiche 5 elements squelettes imitant le layout ConversationItem.
-->
<script lang="ts">
	export let count: number = 5;
</script>

<div class="space-y-1 px-1" role="status" aria-label="Loading conversations">
	{#each Array(count) as _, i}
		<div class="flex items-center gap-2.5 px-3 py-2.5 rounded-lg">
			<!-- Skeleton: icon circle -->
			<div class="w-4 h-4 rounded bg-surface-700 animate-skeleton shrink-0" />
			<!-- Skeleton: text lines -->
			<div class="flex-1 min-w-0 space-y-1.5">
				<div class="h-3 rounded bg-surface-700 animate-skeleton"
					style="width: {70 + (i * 13) % 30}%; max-width: 100%;" />
				<div class="h-2.5 rounded bg-surface-700/60 animate-skeleton"
					style="width: {40 + (i * 17) % 25}%; max-width: 100%;" />
			</div>
		</div>
	{/each}
	<span class="sr-only">Loading conversations...</span>
</div>
