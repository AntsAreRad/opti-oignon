<!--
  SecurityBadge.svelte
  Compact security status badge in the status card. Shows the letter
  grade (A+ to F) with color coding; a tooltip surfaces the numeric score.
  Clicking opens the Workshop's security page, which explains the grade.
  It reads the grade through the API client (lib/api/security.ts), so an
  expired session leads to the sign-in page, as every request does. Uses
  the ds Tooltip and Icon primitives and --oo-* tokens.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import Tooltip from '$lib/ds/Tooltip.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import { getSecurityStatus } from '$lib/api/security';

	let grade = '';
	let score = 0;
	let maxScore = 100;
	let loading = true;

	onMount(async () => {
		try {
			const data = await getSecurityStatus();
			grade = data.grade || '?';
			score = data.score || 0;
			maxScore = data.max_score || 100;
		} catch {
			// Silently fail -- the badge just won't show.
		} finally {
			loading = false;
		}
	});

	function gradeColor(g: string): string {
		if (g.startsWith('A')) return 'var(--oo-success)';
		if (g.startsWith('B')) return 'var(--oo-acc-ink)';
		if (g === 'C') return 'var(--oo-warning)';
		return 'var(--oo-error)';
	}
</script>

{#if !loading && grade}
	<Tooltip content={`Security score: ${score}/${maxScore}`}>
		<a
			href="/workshop/security"
			class="oo-sec-badge"
			aria-label={`Security score ${score} of ${maxScore}, grade ${grade}`}
		>
			<Icon name="shield-check" size="sm" />
			<span class="oo-sec-grade" style="color: {gradeColor(grade)};">{grade}</span>
		</a>
	</Tooltip>
{/if}

<style>
	.oo-sec-badge {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		padding: 2px var(--oo-space-2);
		border-radius: var(--oo-radius-md);
		color: var(--oo-fg-tertiary);
		font-size: var(--oo-text-xs);
		transition: background-color 0.12s ease;
	}
	.oo-sec-badge:hover {
		background-color: var(--oo-bg-surface);
	}
	.oo-sec-grade {
		font-weight: 600;
	}
</style>
