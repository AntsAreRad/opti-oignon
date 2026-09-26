<!--
  SpaceSwitch.svelte
  The one switch between the two spaces: Use (the pages a person works in)
  and Workshop (the operator pages). Switching goes to the space's last
  route, else its home; lib/nav/space.ts decides both, and this control
  holds no rule of its own. The selected segment carries a check, so the
  choice never rests on its fill.

  Compact (the collapsed rail), it is one button that goes to the other
  space, drawn with the Workshop's own mark (the chip), never the one
  Preferences wears just above it.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import Button from '$lib/ds/Button.svelte';
	import IconButton from '$lib/ds/IconButton.svelte';
	import { DESTINATIONS, spaceHome, type Space } from '$lib/nav/destinations';
	import { spaceOf, switchTarget } from '$lib/nav/space';
	import { lastRoutes } from '$lib/stores/lastRoutes';

	export let compact = false;
	/** 44 px targets, for a touch screen. */
	export let large = false;

	const LABELS: Record<Space, string> = { use: 'Use', workshop: 'Workshop' };

	$: current = spaceOf($page.url.pathname) ?? 'use';
	$: other = (current === 'use' ? 'workshop' : 'use') as Space;

	function go(space: Space) {
		if (space === current) return;
		const homes = { use: spaceHome('use', DESTINATIONS), workshop: spaceHome('workshop', DESTINATIONS) };
		goto(switchTarget(space, $lastRoutes, homes));
	}
</script>

{#if compact}
	<IconButton
		icon={other === 'workshop' ? 'chip' : 'chat'}
		label={`Go to ${LABELS[other]}`}
		on:click={() => go(other)}
	/>
{:else}
	<div class="oo-space-switch" role="group" aria-label="Space">
		{#each ['use', 'workshop'] as space}
			<Button
				variant="ghost"
				shape="pill"
				size={large ? 'lg' : 'md'}
				block
				pressed={current === space}
				on:click={() => go(space === 'use' ? 'use' : 'workshop')}
			>
				{LABELS[space === 'use' ? 'use' : 'workshop']}
			</Button>
		{/each}
	</div>
{/if}

<style>
	.oo-space-switch {
		display: flex;
		gap: var(--oo-space-1);
		padding: 3px;
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-bg-subtle);
	}
</style>
