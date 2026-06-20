<!--
  System Status page (S169, route stays /health for compatibility).
  Composes the system status content (overview, module groups, warmup,
  benchmarks, alerts) with the cache management section, on the ds primitives
  and --oo-* tokens.
-->
<script lang="ts">
	import HealthDashboard from '$lib/components/health/HealthDashboard.svelte';
	import CacheManager from '$lib/components/health/CacheManager.svelte';
</script>

<div class="h-full overflow-y-auto">
	<div class="max-w-3xl mx-auto px-4 py-6 flex flex-col gap-8">
		<div>
			<h1 class="text-lg font-semibold" style="color: var(--oo-fg-primary);">System Status</h1>
			<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
				Module health, inference, network, and cache at a glance.
			</p>
		</div>

		<section>
			<HealthDashboard />
		</section>

		<hr style="border: none; border-top: 1px solid var(--oo-bd-subtle);" />

		<section>
			<h2 class="text-base font-medium mb-4" style="color: var(--oo-fg-secondary);">Cache</h2>
			<CacheManager />
		</section>
	</div>
</div>
