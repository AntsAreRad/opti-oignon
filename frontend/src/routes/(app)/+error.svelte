<!--
  A page of either space that failed or does not exist, inside the shell,
  so the sidebar and Stop all stay where they are. An address no page
  serves reaches it through the catch-all page beside it
  ([...missing]/+page.ts), which throws a 404.
-->
<script lang="ts">
	import { page } from '$app/stores';
	import EmptyState from '$lib/ds/EmptyState.svelte';
	import Button from '$lib/ds/Button.svelte';

	$: missing = $page.status === 404;
</script>

<svelte:head>
	<title>{missing ? 'Page not found' : 'Something went wrong'}</title>
</svelte:head>

<div class="oo-error-page">
	<EmptyState
		icon="alert-octagon"
		title={missing ? 'This page does not exist' : 'This page failed to load'}
		description={missing
			? 'The address may be old or mistyped.'
			: ($page.error?.message ?? 'An unexpected error occurred.')}
	>
		<Button variant="secondary" href="/chat">Go to your chats</Button>
	</EmptyState>
</div>

<style>
	.oo-error-page {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
		padding: var(--oo-space-6);
	}
</style>
