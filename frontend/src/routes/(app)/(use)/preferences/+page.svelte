<!--
  Preferences: the reader's own settings (appearance, keyboard, account,
  chats, memory), on the settings hub, whose search reads every group of
  both spaces.
-->
<script lang="ts">
	import SettingsHub from '$lib/components/settings/SettingsHub.svelte';
</script>

<svelte:head>
	<title>Preferences</title>
</svelte:head>

<SettingsHub space="use" />
