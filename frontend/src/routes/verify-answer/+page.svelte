<!--
  routes/verify-answer/+page.svelte
  Redirect: merged into the single /verify tab. Forwards to the manual
  "claim/source pairs" mode.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	onMount(() => {
		goto('/verify?mode=pairs', { replaceState: true });
	});
</script>
