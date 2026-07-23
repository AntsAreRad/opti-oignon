<!--
  PipelineEditor.svelte
  Editeur visuel de pipelines d'execution.
  Permet de creer, modifier, reordonner les etapes d'un pipeline
  base sur les types de l'agentic executor.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { ExecStepInfo, ExecStepTypeInfo } from '$lib/types';

	export let steps: ExecStepInfo[] = [];
	export let stepTypes: ExecStepTypeInfo[] = [];
	export let readOnly: boolean = false;

	const dispatch = createEventDispatcher<{
		change: { steps: ExecStepInfo[] };
	}>();

	let expandedStep: number | null = null;
	let dragIndex: number | null = null;
	let dragOverIndex: number | null = null;

	// Couleurs par type de step
	const STEP_COLORS: Record<string, string> = {
		direct: 'var(--oo-pipe-direct)',
		tools: 'var(--oo-pipe-tools)',
		think: 'var(--oo-pipe-think)',
		think_tools: 'var(--oo-acc-300)',
		web_search: 'var(--oo-pipe-search)',
		code_verify: 'var(--oo-pipe-code)',
		reasoning: 'var(--oo-pipe-reason)',
		consensus: 'var(--oo-pipe-consensus)',
		self_correct: 'var(--oo-pipe-correct)',
	};

	function getStepColor(type: string): string {
		return STEP_COLORS[type] || 'var(--oo-fg-muted)';
	}

	function getStepDescription(type: string): string {
		const found = stepTypes.find(st => st.type === type);
		return found?.description || '';
	}

	function emitChange() {
		dispatch('change', { steps: [...steps] });
	}

	function addStep() {
		const newStep: ExecStepInfo = {
			step_type: 'direct',
			label: 'Step ' + (steps.length + 1),
			model_override: null,
			parameters: {},
			condition: null,
			pass_previous_output: true,
		};
		steps = [...steps, newStep];
		expandedStep = steps.length - 1;
		emitChange();
	}

	function removeStep(index: number) {
		steps = steps.filter((_, i) => i !== index);
		if (expandedStep === index) expandedStep = null;
		else if (expandedStep !== null && expandedStep > index) expandedStep--;
		emitChange();
	}

	function moveStep(index: number, direction: number) {
		const newIdx = index + direction;
		if (newIdx < 0 || newIdx >= steps.length) return;
		const copy = [...steps];
		[copy[index], copy[newIdx]] = [copy[newIdx], copy[index]];
		steps = copy;
		if (expandedStep === index) expandedStep = newIdx;
		else if (expandedStep === newIdx) expandedStep = index;
		emitChange();
	}

	function updateStep(index: number, field: string, value: any) {
		const copy = [...steps];
		copy[index] = { ...copy[index], [field]: value };
		// Auto-label quand on change le type
		if (field === 'step_type' && copy[index].label === steps[index].label) {
			const typeLabel = (value as string).replace(/_/g, ' ');
			copy[index].label = typeLabel.charAt(0).toUpperCase() + typeLabel.slice(1);
		}
		steps = copy;
		emitChange();
	}

	function toggleExpand(index: number) {
		expandedStep = expandedStep === index ? null : index;
	}

	// Drag-and-drop handlers
	function onDragStart(event: DragEvent, index: number) {
		if (readOnly) return;
		dragIndex = index;
		if (event.dataTransfer) {
			event.dataTransfer.effectAllowed = 'move';
			event.dataTransfer.setData('text/plain', String(index));
		}
	}

	function onDragOver(event: DragEvent, index: number) {
		if (readOnly || dragIndex === null) return;
		event.preventDefault();
		dragOverIndex = index;
	}

	function onDragLeave() {
		dragOverIndex = null;
	}

	function onDrop(event: DragEvent, targetIndex: number) {
		if (readOnly || dragIndex === null) return;
		event.preventDefault();
		if (dragIndex !== targetIndex) {
			const copy = [...steps];
			const [removed] = copy.splice(dragIndex, 1);
			copy.splice(targetIndex, 0, removed);
			steps = copy;
			if (expandedStep === dragIndex) expandedStep = targetIndex;
			emitChange();
		}
		dragIndex = null;
		dragOverIndex = null;
	}

	function onDragEnd() {
		dragIndex = null;
		dragOverIndex = null;
	}
</script>

<div class="space-y-1.5">
	{#each steps as step, i (i)}
		<div
			class="rounded-lg overflow-hidden transition-all duration-150"
			style="border: 1px solid {dragOverIndex === i ? getStepColor(step.step_type) : 'var(--oo-bd-default)'}; 
				background-color: var(--oo-bg-surface);
				opacity: {dragIndex === i ? '0.5' : '1'};"
			draggable={!readOnly}
			on:dragstart={(e) => onDragStart(e, i)}
			on:dragover={(e) => onDragOver(e, i)}
			on:dragleave={onDragLeave}
			on:drop={(e) => onDrop(e, i)}
			on:dragend={onDragEnd}
			role="listitem"
		>
			<!-- Step header -->
			<button
				class="w-full flex items-center gap-2 px-3 py-2 text-left transition-colors"
				style="background-color: transparent;"
				on:click={() => toggleExpand(i)}
			>
				<!-- Drag handle -->
				{#if !readOnly}
					<span class="cursor-grab shrink-0" style="color: var(--oo-fg-faint);">
						<svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
							<circle cx="9" cy="6" r="1.5" /><circle cx="15" cy="6" r="1.5" />
							<circle cx="9" cy="12" r="1.5" /><circle cx="15" cy="12" r="1.5" />
							<circle cx="9" cy="18" r="1.5" /><circle cx="15" cy="18" r="1.5" />
						</svg>
					</span>
				{/if}

				<!-- Step number + color indicator -->
				<span
					class="inline-flex items-center justify-center w-5 h-5 rounded-full text-xs font-mono shrink-0"
					style="background-color: {getStepColor(step.step_type)}20; color: {getStepColor(step.step_type)};"
				>
					{i + 1}
				</span>

				<!-- Label and type -->
				<div class="flex-1 min-w-0">
					<div class="text-xs font-medium truncate" style="color: var(--oo-fg-primary);">
						{step.label || 'Unnamed'}
					</div>
					<div class="text-xs truncate" style="color: var(--oo-fg-muted);">
						{step.step_type}
						{#if step.model_override}
							<span style="color: var(--oo-fg-tertiary);"> - {step.model_override}</span>
						{/if}
						{#if step.condition}
							<span style="color: var(--oo-acc-400);"> [{step.condition}]</span>
						{/if}
					</div>
				</div>

				<!-- Connector arrow (sauf dernier) -->
				{#if i < steps.length - 1}
					<span class="shrink-0 text-xs" style="color: var(--oo-fg-faint);">
						<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M19 14l-7 7m0 0l-7-7m7 7V3" />
						</svg>
					</span>
				{/if}

				<!-- Expand chevron -->
				{#if !readOnly}
					<svg
						class="w-3.5 h-3.5 shrink-0 transition-transform"
						class:rotate-180={expandedStep === i}
						style="color: var(--oo-fg-muted);"
						fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"
					>
						<path d="M19 9l-7 7-7-7" />
					</svg>
				{/if}
			</button>

			<!-- Expanded editor -->
			{#if expandedStep === i && !readOnly}
				<div class="px-3 pb-3 space-y-2.5" style="border-top: 1px solid var(--oo-bd-subtle);">
					<!-- Type selector -->
					<div class="mt-2.5">
						<label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">
							Step Type
						</label>
						<select
							value={step.step_type}
							on:change={(e) => updateStep(i, 'step_type', e.currentTarget.value)}
							class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
								border: 1px solid var(--oo-input-bd);"
						>
							{#each stepTypes as st}
								<option value={st.type}>{st.type} - {st.description}</option>
							{/each}
							{#if !stepTypes.find(st => st.type === step.step_type)}
								<option value={step.step_type}>{step.step_type}</option>
							{/if}
						</select>
					</div>

					<!-- Label -->
					<div>
						<label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">
							Label
						</label>
						<input
							type="text"
							value={step.label}
							on:input={(e) => updateStep(i, 'label', e.currentTarget.value)}
							class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
								border: 1px solid var(--oo-input-bd);"
							placeholder="Step label"
						/>
					</div>

					<!-- Model override -->
					<div>
						<label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">
							Model Override (optional)
						</label>
						<input
							type="text"
							value={step.model_override || ''}
							on:input={(e) => updateStep(i, 'model_override', e.currentTarget.value || null)}
							class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none font-mono"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
								border: 1px solid var(--oo-input-bd);"
							placeholder="Leave empty for default"
						/>
					</div>

					<!-- Condition -->
					<div>
						<label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">
							Condition (optional)
						</label>
						<select
							value={step.condition || ''}
							on:change={(e) => updateStep(i, 'condition', e.currentTarget.value || null)}
							class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
								border: 1px solid var(--oo-input-bd);"
						>
							<option value="">Always execute</option>
							<option value="if_code_detected">If code detected</option>
							<option value="if_long_input">If long input (&gt;500 chars)</option>
						</select>
					</div>

					<!-- Pass previous output toggle -->
					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							checked={step.pass_previous_output}
							on:change={(e) => updateStep(i, 'pass_previous_output', e.currentTarget.checked)}
							class="rounded"
							style="accent-color: var(--oo-acc-400);"
						/>
						<span class="text-xs" style="color: var(--oo-fg-tertiary);">
							Pass previous step output
						</span>
					</label>

					<!-- Actions: move / remove -->
					<div class="flex items-center gap-1 pt-1">
						<button
							on:click|stopPropagation={() => moveStep(i, -1)}
							disabled={i === 0}
							class="p-1 rounded transition-colors disabled:opacity-30"
							style="color: var(--oo-fg-tertiary);"
							title="Move up"
						>
							<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M5 15l7-7 7 7" />
							</svg>
						</button>
						<button
							on:click|stopPropagation={() => moveStep(i, 1)}
							disabled={i === steps.length - 1}
							class="p-1 rounded transition-colors disabled:opacity-30"
							style="color: var(--oo-fg-tertiary);"
							title="Move down"
						>
							<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M19 9l-7 7-7-7" />
							</svg>
						</button>
						<div class="flex-1" />
						{#if steps.length > 1}
							<button
								on:click|stopPropagation={() => removeStep(i)}
								class="flex items-center gap-1 text-xs px-2 py-0.5 rounded transition-colors"
								style="color: var(--oo-error);"
								title="Remove step"
							>
								<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
								</svg>
								Remove
							</button>
						{/if}
					</div>
				</div>
			{/if}
		</div>
	{/each}

	<!-- Add step button -->
	{#if !readOnly}
		<button
			on:click={addStep}
			class="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs
				transition-colors"
			style="border: 1px dashed var(--oo-bd-default); color: var(--oo-acc-400);
				background-color: transparent;"
		>
			<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M12 4v16m8-8H4" />
			</svg>
			Add Step
		</button>
	{/if}
</div>
