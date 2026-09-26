<!--
  A Workshop settings page: the settings catalog's groups placed on it,
  named by the last segment of its URL (the parameter matcher is the
  destination table's), on the settings hub.
-->
<script lang="ts">
	import { page } from '$app/stores';
	import SettingsHub from '$lib/components/settings/SettingsHub.svelte';

	$: section = $page.params.section ?? null;
</script>

<SettingsHub space="workshop" {section} />
