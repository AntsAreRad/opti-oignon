<!--
  Notes page (S248).
  Renders NotesPanel inside the shared AppShell (from +layout). The Notes feature
  is a per-user surface over /api/notes (N.2 core UI) with the N.3 selection-action
  panel wired into the editor.
-->
<script lang="ts">
	import NotesPanel from '$lib/components/panels/NotesPanel.svelte';
</script>

<NotesPanel />
