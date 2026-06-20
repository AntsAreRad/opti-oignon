<!--
  routes/claims/+page.svelte
  Redirect: the three verification pages were merged into a single /verify tab.
  This path forwards to the manual "claim/source pairs" mode, which covers the
  former single-claim surface (one pair is just the N=1 case).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	onMount(() => {
		goto('/verify?mode=pairs', { replaceState: true });
	});
</script>
