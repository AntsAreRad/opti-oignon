<!--
  Select.svelte (lib/ds) -- selection primitive (spec 10.7).
  Native <select> for short lists and for multiple (with optgroup
  grouping); a WAI-ARIA combobox (filter + keyboard nav) for searchable
  single lists. Searchable defaults to true when options >= 20.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Icon from './Icon.svelte';
	import type { SelectOption, Size } from './types';

	export let value: string | string[] = '';
	export let multiple = false;
	export let options: SelectOption[] = [];
	export let label: string;
	export let hint: string | undefined = undefined;
	export let error: string | undefined = undefined;
	export let searchable: boolean | undefined = undefined;
	export let placeholder: string | undefined = undefined;
	export let size: Size = 'md';
	export let disabled = false;

	const dispatch = createEventDispatcher<{ change: string | string[] }>();
	const uid = `oo-select-${Math.random().toString(36).slice(2, 9)}`;

	$: isSearchable = (searchable ?? options.length >= 20) && !multiple;
	$: describedBy = error ? `${uid}-error` : hint ? `${uid}-hint` : undefined;

	// Grouping for the native path.
	$: ungrouped = options.filter((o) => !o.group);
	$: groupNames = Array.from(
		options.reduce((set, o) => (o.group ? set.add(o.group) : set), new Set<string>())
	);

	function emit() {
		dispatch('change', value);
	}

	// -- Combobox state --
	let openList = false;
	let query = '';
	let activeIndex = 0;
	let blurTimer: ReturnType<typeof setTimeout> | undefined;

	$: selectedLabel =
		!multiple && typeof value === 'string'
			? options.find((o) => o.value === value)?.label ?? ''
			: '';
	$: filtered = isSearchable
		? options.filter((o) => o.label.toLowerCase().includes(query.toLowerCase()))
		: options;

	function openCombo() {
		openList = true;
		query = '';
		activeIndex = Math.max(
			0,
			filtered.findIndex((o) => o.value === value)
		);
	}

	function selectOption(opt: SelectOption) {
		if (opt.disabled) return;
		value = opt.value;
		query = opt.label;
		openList = false;
		emit();
	}

	function onComboInput(event: Event) {
		query = (event.currentTarget as HTMLInputElement).value;
		openList = true;
		activeIndex = 0;
	}

	function onComboBlur() {
		blurTimer = setTimeout(() => {
			openList = false;
			query = selectedLabel;
		}, 120);
	}

	function onComboFocus() {
		clearTimeout(blurTimer);
		openCombo();
	}

	function onComboKeydown(event: KeyboardEvent) {
		if (event.key === 'ArrowDown') {
			event.preventDefault();
			openList = true;
			activeIndex = Math.min(activeIndex + 1, filtered.length - 1);
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			activeIndex = Math.max(activeIndex - 1, 0);
		} else if (event.key === 'Enter') {
			if (openList && filtered[activeIndex]) {
				event.preventDefault();
				selectOption(filtered[activeIndex]);
			}
		} else if (event.key === 'Escape') {
			openList = false;
			query = selectedLabel;
		}
	}
</script>

<div class="oo-field" data-size={size}>
	<label for={isSearchable ? `${uid}-combo` : uid} class="oo-field-label">{label}</label>

	{#if isSearchable}
		<!-- WAI-ARIA combobox (searchable single) -->
		<div class="oo-combo">
			<!-- svelte-ignore a11y-role-has-required-aria-props -->
			<input
				id={`${uid}-combo`}
				class="oo-field-control"
				data-size={size}
				role="combobox"
				aria-expanded={openList}
				aria-controls={`${uid}-listbox`}
				aria-autocomplete="list"
				aria-activedescendant={openList && filtered[activeIndex]
					? `${uid}-opt-${activeIndex}`
					: undefined}
				aria-invalid={!!error}
				aria-describedby={describedBy}
				placeholder={placeholder ?? selectedLabel}
				{disabled}
				value={openList ? query : selectedLabel}
				on:input={onComboInput}
				on:focus={onComboFocus}
				on:blur={onComboBlur}
				on:keydown={onComboKeydown}
				autocomplete="off"
			/>
			<span class="oo-combo-chevron" aria-hidden="true"><Icon name="chevron-down" size="sm" /></span>
			{#if openList}
				<ul id={`${uid}-listbox`} role="listbox" class="oo-listbox">
					{#each filtered as opt, i (opt.value)}
						<li
							id={`${uid}-opt-${i}`}
							role="option"
							aria-selected={opt.value === value}
							class="oo-option"
							class:active={i === activeIndex}
							class:disabled={opt.disabled}
							on:mousedown|preventDefault={() => selectOption(opt)}
							on:mousemove={() => (activeIndex = i)}
						>
							{opt.label}
						</li>
					{:else}
						<li class="oo-option oo-option-empty" aria-disabled="true">No matches</li>
					{/each}
				</ul>
			{/if}
		</div>
	{:else if multiple}
		<select
			id={uid}
			class="oo-field-control"
			data-size={size}
			multiple
			{disabled}
			aria-invalid={!!error}
			aria-describedby={describedBy}
			bind:value
			on:change={emit}
		>
			{#each ungrouped as opt (opt.value)}
				<option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
			{/each}
			{#each groupNames as g (g)}
				<optgroup label={g}>
					{#each options.filter((o) => o.group === g) as opt (opt.value)}
						<option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
					{/each}
				</optgroup>
			{/each}
		</select>
	{:else}
		<select
			id={uid}
			class="oo-field-control"
			data-size={size}
			{disabled}
			aria-invalid={!!error}
			aria-describedby={describedBy}
			bind:value
			on:change={emit}
		>
			{#if placeholder}
				<option value="" disabled selected={value === ''}>{placeholder}</option>
			{/if}
			{#each ungrouped as opt (opt.value)}
				<option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
			{/each}
			{#each groupNames as g (g)}
				<optgroup label={g}>
					{#each options.filter((o) => o.group === g) as opt (opt.value)}
						<option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
					{/each}
				</optgroup>
			{/each}
		</select>
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
	.oo-field-control {
		width: 100%;
		font-family: var(--oo-font-sans);
		color: var(--oo-fg-primary);
		background-color: var(--oo-bg-input);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
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
	.oo-field-control:focus {
		outline: none;
		border-color: var(--oo-acc-500);
		box-shadow: 0 0 0 3px var(--oo-input-focus);
	}
	.oo-field-control[aria-invalid='true'] {
		border-color: var(--oo-error);
	}
	.oo-combo {
		position: relative;
	}
	.oo-combo-chevron {
		position: absolute;
		right: var(--oo-space-3);
		top: 50%;
		transform: translateY(-50%);
		color: var(--oo-fg-muted);
		pointer-events: none;
	}
	.oo-listbox {
		position: absolute;
		z-index: var(--oo-z-overlay);
		top: calc(100% + 4px);
		left: 0;
		right: 0;
		margin: 0;
		padding: var(--oo-space-1);
		list-style: none;
		max-height: 16rem;
		overflow-y: auto;
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		box-shadow: var(--oo-shadow-md);
	}
	.oo-option {
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-sm);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
		cursor: pointer;
	}
	.oo-option.active {
		background-color: var(--oo-bg-hover);
	}
	.oo-option[aria-selected='true'] {
		color: var(--oo-acc-500);
		font-weight: 500;
	}
	.oo-option.disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.oo-option-empty {
		color: var(--oo-fg-muted);
		cursor: default;
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
</style>
