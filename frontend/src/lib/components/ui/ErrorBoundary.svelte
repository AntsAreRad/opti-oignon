<!--
  ErrorBoundary.svelte
  Wraps child content and captures render errors.
  Displays a friendly error message with retry button and optional report link.
  Usage: <ErrorBoundary><slot /></ErrorBoundary>
-->
<script lang="ts">
	import { onMount } from 'svelte';

	export let fallbackMessage: string = 'Something went wrong';

	let hasError = false;
	let errorMessage = '';

	function handleError(event: ErrorEvent) {
		hasError = true;
		errorMessage = event.message || 'An unexpected error occurred';
		// Log error to console for debugging
		console.error('[ErrorBoundary]', event.message, event.filename, event.lineno);
		event.preventDefault();
	}

	function retry() {
		hasError = false;
		errorMessage = '';
	}

	onMount(() => {
		// Capture unhandled errors in the subtree
		window.addEventListener('error', handleError);
		return () => window.removeEventListener('error', handleError);
	});
</script>

{#if hasError}
	<div class="flex flex-col items-center justify-center p-8 text-center" role="alert" aria-live="assertive">
		<div class="w-12 h-12 rounded-full flex items-center justify-center mb-4"
			style="background-color: var(--oo-error-bg);">
			<svg class="w-6 h-6" style="color: var(--oo-error);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v4m0 4h.01M12 2l10 18H2L12 2z" />
			</svg>
		</div>
		<h3 class="text-sm font-medium mb-1" style="color: var(--oo-fg-secondary);">{fallbackMessage}</h3>
		{#if errorMessage}
			<p class="text-xs mb-4 max-w-sm" style="color: var(--oo-fg-muted);">{errorMessage}</p>
		{/if}
		<div class="flex items-center gap-3">
			<button
				on:click={retry}
				class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
				style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-secondary);
					border: 1px solid var(--oo-bd-default);"
				aria-label="Retry loading content"
			>
				<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M1 4v6h6" />
					<path d="M3.51 15a9 9 0 102.13-9.36L1 10" />
				</svg>
				Try again
			</button>
			<a
				href="https://github.com/anthropics/opti-oignon/issues"
				target="_blank"
				rel="noopener noreferrer"
				class="text-xs underline transition-colors"
				style="color: var(--oo-fg-muted);"
				aria-label="Report this issue on GitHub"
			>
				Report issue
			</a>
		</div>
	</div>
{:else}
	<slot />
{/if}
