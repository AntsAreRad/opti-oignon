<!--
  SandboxIsolationBadge.svelte
  Shows the active sandbox isolation backend (bwrap / tempdir / none) as a
  small inline badge. Green for bwrap, amber for tempdir, red for none.
  Uses the ds Icon primitive and --oo-* tokens (no raw hex).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import { getSandboxStatus } from '$lib/api/sandbox';

	let backend = '';
	let loaded = false;

	onMount(async () => {
		try {
			const status = await getSandboxStatus();
			backend = status.isolation_backend || 'unknown';
		} catch {
			// Silently fail.
		} finally {
			loaded = true;
		}
	});

	$: badgeColor =
		backend === 'bwrap'
			? 'var(--oo-success)'
			: backend === 'tempdir'
				? 'var(--oo-warning)'
				: 'var(--oo-error)';

	$: label = backend === 'bwrap' ? 'Isolated' : backend === 'tempdir' ? 'Partial' : 'No isolation';

	$: tooltip =
		backend === 'bwrap'
			? 'Bubblewrap sandbox active (fully isolated)'
			: backend === 'tempdir'
				? 'Tempdir fallback (limited isolation)'
				: 'No sandbox isolation available';
</script>

{#if loaded && backend}
	<span
		class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs"
		style="color: {badgeColor}; border: 1px solid {badgeColor}; opacity: 0.85;"
		title={tooltip}
	>
		{#if backend === 'bwrap'}
			<Icon name="shield-check" size="sm" />
		{:else}
			<Icon name="alert-triangle" size="sm" />
		{/if}
		{label}
	</span>
{/if}
