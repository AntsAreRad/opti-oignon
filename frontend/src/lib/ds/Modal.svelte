<!--
  Modal.svelte (lib/ds) -- single modal primitive, three variants
  (spec 10.5). Replaces the seven independent `fixed inset-0`
  implementations. Renders inside a native <dialog> via showModal()
  for a built-in focus trap; ESC and backdrop clicks route through the
  `open` state and `onClose`. drawer-right becomes drawer-bottom on
  mobile (< 768px). Focus is restored to the opener on close.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Icon from './Icon.svelte';
	import type { ModalVariant } from './types';

	export let open = false;
	export let variant: ModalVariant = 'center';
	export let title: string;
	export let size: 'sm' | 'md' | 'lg' | 'xl' = 'md';
	export let closeOnBackdrop = true;
	export let closeOnEsc = true;
	export let onClose: () => void;

	const uid = `oo-modal-${Math.random().toString(36).slice(2, 9)}`;
	let dialogEl: HTMLDialogElement;
	let opener: HTMLElement | null = null;
	let isMobile = false;

	$: effectiveVariant =
		variant === 'drawer-right' && isMobile ? 'drawer-bottom' : variant;

	onMount(() => {
		const mq = window.matchMedia('(max-width: 767px)');
		const apply = () => (isMobile = mq.matches);
		apply();
		mq.addEventListener('change', apply);
		return () => mq.removeEventListener('change', apply);
	});

	async function openDialog() {
		opener = (document.activeElement as HTMLElement) ?? null;
		if (!dialogEl.open) dialogEl.showModal();
		await tick();
		const focusable = dialogEl.querySelector<HTMLElement>(
			'[autofocus], button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
		);
		focusable?.focus();
	}

	function closeDialog() {
		if (dialogEl.open) dialogEl.close();
		opener?.focus?.();
		opener = null;
	}

	$: if (dialogEl) {
		if (open && !dialogEl.open) openDialog();
		else if (!open && dialogEl.open) closeDialog();
	}

	function requestClose() {
		open = false;
		onClose();
	}

	function onCancel(event: Event) {
		// ESC fires the native cancel event; route it through our state.
		event.preventDefault();
		if (closeOnEsc) requestClose();
	}

	function onDialogClick(event: MouseEvent) {
		if (closeOnBackdrop && event.target === dialogEl) requestClose();
	}
</script>

<dialog
	bind:this={dialogEl}
	class="oo-modal"
	data-variant={effectiveVariant}
	data-size={size}
	aria-modal="true"
	aria-labelledby={`${uid}-title`}
	on:cancel={onCancel}
	on:click={onDialogClick}
>
	<div class="oo-modal-panel" data-variant={effectiveVariant} data-size={size}>
		<header class="oo-modal-header">
			<h2 id={`${uid}-title`} class="oo-modal-title">{title}</h2>
			<button
				type="button"
				class="oo-modal-close"
				aria-label="Close dialog"
				on:click={requestClose}
			>
				<Icon name="x" size="sm" />
			</button>
		</header>

		<div class="oo-modal-body">
			<slot />
		</div>

		{#if $$slots.footer}
			<footer class="oo-modal-footer">
				<slot name="footer" />
			</footer>
		{/if}
	</div>
</dialog>

<style>
	.oo-modal {
		padding: 0;
		border: none;
		background: transparent;
		max-width: 100vw;
		max-height: 100vh;
		color: var(--oo-fg-primary);
	}
	.oo-modal::backdrop {
		background: rgba(0, 0, 0, 0.5);
	}

	.oo-modal-panel {
		display: flex;
		flex-direction: column;
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		box-shadow: var(--oo-shadow-lg);
		max-height: 90vh;
		overflow: hidden;
	}

	/* Center variant */
	.oo-modal[data-variant='center'] {
		margin: auto;
	}
	.oo-modal-panel[data-variant='center'] {
		border-radius: var(--oo-radius-xl);
		width: min(92vw, var(--oo-modal-w, 32rem));
	}
	.oo-modal-panel[data-variant='center'][data-size='sm'] {
		--oo-modal-w: 24rem;
	}
	.oo-modal-panel[data-variant='center'][data-size='md'] {
		--oo-modal-w: 32rem;
	}
	.oo-modal-panel[data-variant='center'][data-size='lg'] {
		--oo-modal-w: 44rem;
	}
	.oo-modal-panel[data-variant='center'][data-size='xl'] {
		--oo-modal-w: 60rem;
	}

	/* Drawer-right */
	.oo-modal[data-variant='drawer-right'] {
		margin: 0 0 0 auto;
		height: 100vh;
	}
	.oo-modal-panel[data-variant='drawer-right'] {
		height: 100vh;
		max-height: 100vh;
		border-radius: 0;
		width: min(92vw, var(--oo-modal-w, 28rem));
	}
	.oo-modal-panel[data-variant='drawer-right'][data-size='lg'] {
		--oo-modal-w: 40rem;
	}
	.oo-modal-panel[data-variant='drawer-right'][data-size='xl'] {
		--oo-modal-w: 52rem;
	}

	/* Drawer-bottom */
	.oo-modal[data-variant='drawer-bottom'] {
		margin: auto auto 0 auto;
		width: 100vw;
	}
	.oo-modal-panel[data-variant='drawer-bottom'] {
		width: 100vw;
		border-radius: var(--oo-radius-xl) var(--oo-radius-xl) 0 0;
		max-height: 85vh;
	}

	.oo-modal-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-4);
		padding: var(--oo-space-5);
		border-bottom: 1px solid var(--oo-bd-subtle);
	}
	.oo-modal-title {
		margin: 0;
		font-size: var(--oo-text-lg);
		font-weight: 600;
		color: var(--oo-fg-primary);
	}
	.oo-modal-close {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		border: none;
		border-radius: var(--oo-radius-md);
		background: transparent;
		color: var(--oo-fg-secondary);
		cursor: pointer;
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-modal-close:hover {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}
	.oo-modal-body {
		padding: var(--oo-space-5);
		overflow-y: auto;
	}
	.oo-modal-footer {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: var(--oo-space-3);
		padding: var(--oo-space-5);
		border-top: 1px solid var(--oo-bd-subtle);
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-modal-close {
			transition: none;
		}
	}
</style>
