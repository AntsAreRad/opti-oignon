<!--
  PhoneHeader.svelte
  The shell's header on a phone (under 768 px), outside the drawer: the
  drawer's opener, the page's title, and Stop all, so the stop is one tap
  away on every page while the drawer is shut (open, the drawer holds its
  own). The count of the tool calls waiting on an approval appears beside
  the stop, in its compact form, so the two fit a narrow screen; the title
  gives way first. Every control is a 44 px target, the stop's
  confirmation and Resume included, and the header keeps clear of the notch.

  It sits above every layer that is not modal (a side panel, a menu, the
  page's own raised parts), so nothing but a dialog covers the stop.
-->
<script lang="ts">
	import IconButton from '$lib/ds/IconButton.svelte';
	import StopAllButton from './StopAllButton.svelte';
	import ApprovalsPill from './ApprovalsPill.svelte';

	/** The page's name. */
	export let title: string;
	/** Whether the drawer is open. */
	export let open: boolean;
	/** The id of the drawer the opener controls. */
	export let drawer: string;
	export let onToggle: () => void = () => {};
</script>

<header class="oo-phone-header">
	<IconButton
		icon="panel-left"
		size="lg"
		label={open ? 'Close navigation' : 'Open navigation'}
		expanded={open}
		controls={drawer}
		on:click={onToggle}
	/>
	<span class="oo-phone-title">{title}</span>
	<ApprovalsPill compact large />
	<StopAllButton placement="header" />
</header>

<style>
	.oo-phone-header {
		position: relative;
		z-index: calc(var(--oo-z-overlay) + 10);
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		min-height: 56px;
		padding: calc(var(--oo-space-2) + env(safe-area-inset-top, 0px)) calc(var(--oo-space-3) + env(safe-area-inset-right, 0px)) var(--oo-space-2) calc(var(--oo-space-2) + env(safe-area-inset-left, 0px));
		background-color: var(--oo-bg-base);
		color: var(--oo-fg-primary);
	}
	.oo-phone-title {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-lg);
		font-weight: 600;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
