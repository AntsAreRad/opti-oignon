<!--
  StatusCard.svelte
  The status card at the foot of the sidebar: the machine at a glance, with
  the stop beside it.

  The first row names the inference backend and its state, then holds Stop
  all. Two failures read apart: "Server unreachable" when the API itself
  did not answer, and "<backend> unavailable" when the server answered and
  its backend is down. The state is the backend status store's
  (lib/stores/backendStatus.ts), read once a minute while the page is
  visible; the card requests nothing itself.

  A second, quieter row holds the security grade (leading to the Workshop's
  security page), the mode word in Bulbe alone (Daily is the ordinary mode
  and says nothing), and a pill while tool calls wait on an approval.

  In the phone's drawer (`large`) every control and link is a 44 px target.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import SecurityBadge from '$lib/components/sidebar/SecurityBadge.svelte';
	import StopAllButton from './StopAllButton.svelte';
	import ApprovalsPill from './ApprovalsPill.svelte';
	import { backendStatus } from '$lib/stores/backendStatus';
	import { currentMode, refreshSecurityMode } from '$lib/stores/securityMode';

	/** The drawer's card on a phone: its controls are 44 px targets. */
	export let large = false;

	type Health = 'checking' | 'ok' | 'down' | 'none';

	$: name = $backendStatus.backend?.display_name || $backendStatus.backend?.name || '';
	$: health = ((): Health => {
		if ($backendStatus.reachable === null) return 'checking';
		if ($backendStatus.reachable === false) return 'down';
		if (!$backendStatus.backend) return 'none';
		return $backendStatus.backend.healthy ? 'ok' : 'down';
	})();
	$: headline =
		$backendStatus.reachable === false
			? 'Server unreachable'
			: health === 'checking'
				? 'Checking the server'
				: health === 'none'
					? 'No inference backend'
					: health === 'down'
						? `${name} unavailable`
						: name;
	$: detail = health === 'ok' ? 'Connected' : '';

	onMount(() => {
		void refreshSecurityMode();
	});
</script>

<section class="oo-card" aria-label="Machine status" data-large={large ? 'true' : 'false'}>
	<div class="oo-card-main">
		<span class="oo-card-dot" data-health={health} aria-hidden="true"></span>
		<span class="oo-card-text">
			<span class="oo-card-headline">{headline}</span>
			{#if detail}
				<span class="oo-card-detail">{detail}</span>
			{/if}
		</span>
		<StopAllButton placement="card" {large} />
	</div>
	<div class="oo-card-quiet">
		<SecurityBadge />
		{#if $currentMode === 'bulbe'}
			<a class="oo-card-mode" href="/workshop/security">Bulbe mode</a>
		{/if}
		<ApprovalsPill {large} />
	</div>
</section>

<style>
	.oo-card {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		padding: var(--oo-space-3) var(--oo-space-3) var(--oo-space-3) var(--oo-space-4);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-2xl);
		background-color: var(--oo-bg-surface);
	}

	.oo-card-main {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		min-width: 0;
	}

	.oo-card-dot {
		width: 8px;
		height: 8px;
		flex-shrink: 0;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-fg-muted);
	}
	.oo-card-dot[data-health='ok'] {
		background-color: var(--oo-status-ok);
	}
	.oo-card-dot[data-health='down'],
	.oo-card-dot[data-health='none'] {
		background-color: var(--oo-fg-stop);
	}

	.oo-card-text {
		display: flex;
		flex: 1;
		flex-direction: column;
		min-width: 0;
		font-size: var(--oo-text-sm);
		line-height: var(--oo-leading-snug);
	}
	.oo-card-headline {
		overflow: hidden;
		color: var(--oo-fg-primary);
		font-weight: 500;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.oo-card-detail {
		color: var(--oo-fg-muted);
	}

	.oo-card-quiet {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-2);
		min-height: 24px;
		font-size: var(--oo-text-xs);
	}
	.oo-card-quiet:empty {
		display: none;
	}

	.oo-card-mode {
		color: var(--oo-acc-ink-2);
		font-weight: 600;
		text-decoration: none;
	}
	.oo-card-mode:hover {
		text-decoration: underline;
	}
	.oo-card[data-large='true'] .oo-card-mode,
	.oo-card[data-large='true'] :global(.oo-sec-badge) {
		display: inline-flex;
		align-items: center;
		min-height: 44px;
	}
</style>
