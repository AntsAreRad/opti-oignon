<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	onMount(() => {
		goto('/chat', { replaceState: true });
	});
</script>

<div class="h-screen flex items-center justify-center bg-surface-900">
	<div class="w-5 h-5 border-2 border-surface-600 border-t-accent-500 rounded-full animate-spin" />
</div>
