<!--
  Input.svelte (lib/ds) -- form field primitive (spec 10.3).
  Types: text, email, password, number, textarea. Label is always
  provided (visually hidden via oo-sr-only when hideLabel). Error is
  wired through aria-describedby + aria-invalid. Two-way bound `value`.

  Note: the input uses an explicit value + on:input handler (rather than
  bind:value) so the `type` attribute can stay dynamic, which Svelte
  forbids together with two-way binding.
-->
<script lang="ts">
	import Icon from './Icon.svelte';
	import type { IconName, Size } from './types';

	export let type: 'text' | 'email' | 'password' | 'number' | 'textarea' = 'text';
	export let value: string | number = '';
	export let label: string;
	export let hint: string | undefined = undefined;
	export let error: string | undefined = undefined;
	export let size: Size = 'md';
	export let iconLeft: IconName | undefined = undefined;
	export let iconRight: IconName | undefined = undefined;
	export let disabled = false;
	export let required = false;
	export let placeholder: string | undefined = undefined;
	export let autocomplete: string | undefined = undefined;
	export let hideLabel = false;
	export let rows = 3;

	const uid = `oo-input-${Math.random().toString(36).slice(2, 9)}`;
	$: describedBy = error ? `${uid}-error` : hint ? `${uid}-hint` : undefined;

	function onInput(event: Event) {
		const target = event.currentTarget as HTMLInputElement | HTMLTextAreaElement;
		if (type === 'number') {
			value = target.value === '' ? '' : Number(target.value);
		} else {
			value = target.value;
		}
	}
</script>

<div class="oo-field" data-size={size}>
	<label for={uid} class="oo-field-label" class:oo-sr-only={hideLabel}>
		{label}{#if required}<span class="oo-field-req" aria-hidden="true"> *</span>{/if}
	</label>

	{#if type === 'textarea'}
		<textarea
			id={uid}
			class="oo-field-control oo-field-textarea"
			data-size={size}
			{rows}
			{placeholder}
			{disabled}
			{required}
			aria-invalid={!!error}
			aria-describedby={describedBy}
			value={String(value)}
			on:input={onInput}
			on:change
			on:focus
			on:blur
			on:keydown
		></textarea>
	{:else}
		<div class="oo-field-wrap" class:has-left={!!iconLeft} class:has-right={!!iconRight}>
			{#if iconLeft}
				<span class="oo-field-icon oo-field-icon-left" aria-hidden="true">
					<Icon name={iconLeft} size="sm" />
				</span>
			{/if}
			<input
				id={uid}
				type={type}
				class="oo-field-control"
				data-size={size}
				{placeholder}
				{disabled}
				{required}
				autocomplete={autocomplete}
				aria-invalid={!!error}
				aria-describedby={describedBy}
				value={String(value)}
				on:input={onInput}
				on:change
				on:focus
				on:blur
				on:keydown
			/>
			{#if iconRight}
				<span class="oo-field-icon oo-field-icon-right" aria-hidden="true">
					<Icon name={iconRight} size="sm" />
				</span>
			{/if}
		</div>
	{/if}

	{#if error}
		<p id={`${uid}-error`} class="oo-field-error">{error}</p>
	{:else if hint}
		<p id={`${uid}-hint`} class="oo-field-hint">{hint}</p>
	{/if}
</div>

<style>
	.oo-field {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}
	.oo-field-label {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-primary);
	}
	.oo-field-req {
		color: var(--oo-error);
	}
	.oo-field-wrap {
		position: relative;
		display: flex;
		align-items: center;
	}
	.oo-field-control {
		width: 100%;
		font-family: var(--oo-font-sans);
		color: var(--oo-fg-primary);
		background-color: var(--oo-bg-input);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		transition:
			border-color var(--oo-motion-fast) var(--oo-ease-default),
			box-shadow var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-field-control[data-size='sm'] {
		font-size: var(--oo-text-xs);
		padding: var(--oo-space-2) var(--oo-space-3);
	}
	.oo-field-control[data-size='md'] {
		font-size: var(--oo-text-sm);
		padding: var(--oo-space-3) var(--oo-space-4);
	}
	.oo-field-control[data-size='lg'] {
		font-size: var(--oo-text-base);
		padding: var(--oo-space-4) var(--oo-space-5);
	}
	.oo-field-textarea {
		resize: vertical;
		min-height: 4rem;
		line-height: var(--oo-leading-normal);
	}
	.oo-field-control::placeholder {
		color: var(--oo-fg-muted);
	}
	.oo-field-control:focus {
		outline: none;
		border-color: var(--oo-acc-500);
		box-shadow: 0 0 0 3px var(--oo-input-focus);
	}
	.oo-field-control:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
	.oo-field-control[aria-invalid='true'] {
		border-color: var(--oo-error);
	}
	.oo-field-wrap.has-left .oo-field-control {
		padding-left: calc(var(--oo-space-4) + 18px);
	}
	.oo-field-wrap.has-right .oo-field-control {
		padding-right: calc(var(--oo-space-4) + 18px);
	}
	.oo-field-icon {
		position: absolute;
		display: inline-flex;
		color: var(--oo-fg-muted);
		pointer-events: none;
	}
	.oo-field-icon-left {
		left: var(--oo-space-3);
	}
	.oo-field-icon-right {
		right: var(--oo-space-3);
	}
	.oo-field-hint {
		margin: 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
	}
	.oo-field-error {
		margin: 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-error);
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-field-control {
			transition: none;
		}
	}
</style>
