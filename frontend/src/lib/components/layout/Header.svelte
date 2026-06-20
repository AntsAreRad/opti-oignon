<!--
  Header.svelte (S167)
  Consolidated header status cluster, mounted inside the AppShell
  <header> landmark (spec 8.4): backend status, theme control,
  notifications and user menu, right-aligned. Phase 2 extracts the
  cluster from AppShell as-is; Phase 3 swaps the binary toggle for the
  ThemeSwitcher and merges NetworkIndicator into BackendStatus.
  S215 prepends the EmergencyStopControl (panic control + resume),
  visible app-wide because this cluster renders on every route.
-->
<script lang="ts">
	import EmergencyStopControl from '$lib/components/ui/EmergencyStopControl.svelte';
	import BackendStatus from '$lib/components/ui/BackendStatus.svelte';
	import ThemeSwitcher from '$lib/components/ui/ThemeSwitcher.svelte';
	import NotificationCenter from '$lib/components/ui/NotificationCenter.svelte';
	import UserMenu from '$lib/components/ui/UserMenu.svelte';
</script>

<div class="oo-header-cluster">
	<EmergencyStopControl />
	<BackendStatus />
	<ThemeSwitcher />
	<NotificationCenter />
	<UserMenu />
</div>

<style>
	.oo-header-cluster {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
</style>
