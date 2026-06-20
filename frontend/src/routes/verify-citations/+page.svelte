<!--
  routes/verify-citations/+page.svelte
  Redirect: merged into the single /verify tab. Forwards to the "cited answer"
  mode (paste an answer with [n] markers plus sources; pairs are auto-extracted).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	onMount(() => {
		goto('/verify?mode=cited', { replaceState: true });
	});
</script>
