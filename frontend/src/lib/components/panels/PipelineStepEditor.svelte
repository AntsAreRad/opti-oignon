<!--
  PipelineStepEditor.svelte
  Editeur d'une etape de pipeline: nom, agent, modele, prompt template.
  Composant enfant de PipelinePanel.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { PipelineStepInfo } from '$lib/types';

	export let step: PipelineStepInfo;
	export let index: number;
	export let agents: string[] = [];
	export let removable: boolean = true;

	const dispatch = createEventDispatcher<{
		update: { index: number; step: PipelineStepInfo };
		remove: { index: number };
	}>();

	let expanded = false;

	function updateField<K extends keyof PipelineStepInfo>(field: K, value: PipelineStepInfo[K]) {
		dispatch('update', { index, step: { ...step, [field]: value } });
	}

	function handleRemove() {
		dispatch('remove', { index });
	}
</script>

<div class="border border-surface-700/50 rounded-lg bg-surface-900/30 overflow-hidden">
	<!-- Step header (toujours visible) -->
	<button
		on:click={() => { expanded = !expanded; }}
		class="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-800/30 transition-colors"
	>
		<!-- Numero d'etape -->
		<span class="inline-flex items-center justify-center w-5 h-5 rounded-full bg-accent-600/20 text-accent-400 text-xs font-mono shrink-0">
			{index + 1}
		</span>

		<!-- Nom et agent -->
		<div class="flex-1 min-w-0">
			<div class="text-xs font-medium text-surface-200 truncate">{step.name || 'Unnamed step'}</div>
			<div class="text-xs text-surface-500">{step.agent}</div>
		</div>

		<!-- Modele badge -->
		{#if step.model}
			<span class="text-xs text-surface-400 font-mono bg-surface-800 px-1.5 py-0.5 rounded shrink-0">
				{step.model}
			</span>
		{/if}

		<!-- Chevron -->
		<svg
			class="w-3.5 h-3.5 text-surface-500 shrink-0 transition-transform"
			class:rotate-180={expanded}
			fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"
		>
			<path d="M19 9l-7 7-7-7" />
		</svg>
	</button>

	<!-- Step details (expanded) -->
	{#if expanded}
		<div class="px-3 pb-3 space-y-2.5 border-t border-surface-700/30">
			<!-- Nom -->
			<div class="mt-2.5">
				<label class="block text-xs text-surface-400 mb-1" for="step-name-{index}">Name</label>
				<input
					id="step-name-{index}"
					type="text"
					value={step.name}
					on:input={(e) => updateField('name', e.currentTarget.value)}
					class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
						text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
					placeholder="Step name"
				/>
			</div>

			<!-- Agent -->
			<div>
				<label class="block text-xs text-surface-400 mb-1" for="step-agent-{index}">Agent</label>
				{#if agents.length > 0}
					<select
						id="step-agent-{index}"
						value={step.agent}
						on:change={(e) => updateField('agent', e.currentTarget.value)}
						class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
							text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
					>
						{#each agents as agent}
							<option value={agent}>{agent}</option>
						{/each}
						{#if !agents.includes(step.agent)}
							<option value={step.agent}>{step.agent}</option>
						{/if}
					</select>
				{:else}
					<input
						id="step-agent-{index}"
						type="text"
						value={step.agent}
						on:input={(e) => updateField('agent', e.currentTarget.value)}
						class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
							text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
						placeholder="Agent type"
					/>
				{/if}
			</div>

			<!-- Modele -->
			<div>
				<label class="block text-xs text-surface-400 mb-1" for="step-model-{index}">Model (optional)</label>
				<input
					id="step-model-{index}"
					type="text"
					value={step.model || ''}
					on:input={(e) => updateField('model', e.currentTarget.value || null)}
					class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
						text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
					placeholder="Default model (leave empty for auto)"
				/>
			</div>

			<!-- Description -->
			<div>
				<label class="block text-xs text-surface-400 mb-1" for="step-desc-{index}">Description</label>
				<input
					id="step-desc-{index}"
					type="text"
					value={step.description}
					on:input={(e) => updateField('description', e.currentTarget.value)}
					class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
						text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
					placeholder="Step description"
				/>
			</div>

			<!-- Prompt template -->
			<div>
				<label class="block text-xs text-surface-400 mb-1" for="step-prompt-{index}">Prompt template</label>
				<textarea
					id="step-prompt-{index}"
					value={step.prompt_template || ''}
					on:input={(e) => updateField('prompt_template', e.currentTarget.value || null)}
					class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
						text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500
						resize-y min-h-[60px] max-h-[200px] font-mono"
					rows="3"
					placeholder="Prompt template with &#123;input&#125; placeholder"
				></textarea>
			</div>

			<!-- System prompt -->
			<div>
				<label class="block text-xs text-surface-400 mb-1" for="step-sys-{index}">System prompt</label>
				<textarea
					id="step-sys-{index}"
					value={step.system_prompt || ''}
					on:input={(e) => updateField('system_prompt', e.currentTarget.value || null)}
					class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
						text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500
						resize-y min-h-[60px] max-h-[200px]"
					rows="2"
					placeholder="System prompt override"
				></textarea>
			</div>

			<!-- Supprimer l'etape -->
			{#if removable}
				<button
					on:click={handleRemove}
					class="flex items-center gap-1 text-xs text-[var(--oo-error)] hover:text-[var(--oo-error)] transition-colors mt-1"
				>
					<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
					</svg>
					Remove step
				</button>
			{/if}
		</div>
	{/if}
</div>
