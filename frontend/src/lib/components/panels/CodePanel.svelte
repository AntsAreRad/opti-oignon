<!--
  CodePanel.svelte
  Panneau d'execution de code: editeur, selecteur de langage,
  bouton execute, affichage stdout/stderr, duree.
-->
<script lang="ts">
	import { closePanel } from '$lib/stores/panels';
	import { messages } from '$lib/stores/conversations';
	import { toastError, toastSuccess } from '$lib/stores/notifications';
	import { executeCode, extractCodeBlocks, resetWorkdir } from '$lib/api/code';
	import type { CodeBlockInfo, CodeExecuteResponse } from '$lib/types';

	let code = '';
	let language = 'python';
	let executing = false;
	let result: CodeExecuteResponse | null = null;
	let extractedBlocks: CodeBlockInfo[] = [];
	let showExtracted = false;

	const languages = ['python', 'javascript', 'bash', 'r', 'sql'];

	async function handleExecute() {
		if (!code.trim()) return;
		executing = true;
		result = null;
		try {
			result = await executeCode({ code, language });
		} catch (err) {
			toastError('Code execution failed');
		} finally {
			executing = false;
		}
	}

	async function handleReset() {
		try {
			await resetWorkdir();
			result = null;
			toastSuccess('Workdir reset');
		} catch {
			toastError('Failed to reset workdir');
		}
	}

	async function handleExtract() {
		// Trouve le dernier message assistant
		const lastAssistant = [...$messages].reverse().find((m) => m.role === 'assistant');
		if (!lastAssistant) {
			toastError('No assistant message found');
			return;
		}
		try {
			extractedBlocks = await extractCodeBlocks(lastAssistant.content);
			showExtracted = extractedBlocks.length > 0;
			if (extractedBlocks.length === 0) {
				toastError('No code blocks found');
			}
		} catch {
			toastError('Failed to extract code blocks');
		}
	}

	function loadBlock(block: CodeBlockInfo) {
		code = block.code;
		language = block.language || 'python';
		showExtracted = false;
	}

	function handleKeydown(event: KeyboardEvent) {
		// Ctrl+Enter ou Cmd+Enter pour executer
		if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
			event.preventDefault();
			handleExecute();
		}
	}
</script>

<div class="flex flex-col h-full" style="background-color: var(--oo-panel-bg);">
	<!-- Header -->
	<div class="flex items-center justify-between px-3 py-2 shrink-0" style="border-bottom: 1px solid var(--oo-bd-subtle);">
		<h2 class="text-sm font-medium text-surface-300">Code Execution</h2>
		<div class="flex items-center gap-1">
			<button
				on:click={handleReset}
				class="px-2 py-0.5 text-xs rounded text-surface-400 hover:text-surface-200 hover:bg-surface-800 transition-colors"
				title="Reset working directory"
			>
				Reset
			</button>
			<button
				on:click={closePanel}
				class="p-1 rounded text-surface-400 hover:text-surface-200 hover:bg-surface-800"
				title="Close panel"
			aria-label="Close code panel"
			>
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M6 18L18 6M6 6l12 12" />
				</svg>
			</button>
		</div>
	</div>

	<!-- Code input area -->
	<div class="flex flex-col flex-1 min-h-0">
		<!-- Language + actions bar -->
		<div class="flex items-center gap-2 px-3 py-1.5 border-b border-surface-800/50 shrink-0">
			<select
				bind:value={language}
				class="text-xs bg-surface-900 border border-surface-700 rounded px-2 py-1 text-surface-300 focus:outline-none focus:border-accent-500"
			>
				{#each languages as lang}
					<option value={lang}>{lang}</option>
				{/each}
			</select>

			<button
				on:click={handleExtract}
				class="text-xs text-surface-400 hover:text-surface-200 px-2 py-1 rounded hover:bg-surface-800 transition-colors"
				title="Extract code blocks from last assistant message"
			>
				Extract from chat
			</button>

			<div class="flex-1" />

			<span class="text-xs text-surface-600 hidden sm:inline">Ctrl+Enter to run</span>
		</div>

		<!-- Extracted blocks dropdown -->
		{#if showExtracted && extractedBlocks.length > 0}
			<div class="border-b border-surface-800/50 bg-surface-900/50 max-h-36 overflow-y-auto">
				{#each extractedBlocks as block, i}
					<button
						on:click={() => loadBlock(block)}
						class="w-full text-left px-3 py-1.5 hover:bg-surface-800 transition-colors border-b border-surface-800/30 last:border-0"
					>
						<span class="text-xs text-accent-400 font-mono">{block.language || 'code'}</span>
						<span class="text-xs text-surface-500 ml-2">Block {i + 1}</span>
						<pre class="text-xs text-surface-400 mt-0.5 truncate font-mono">{block.code.slice(0, 80)}{block.code.length > 80 ? '...' : ''}</pre>
					</button>
				{/each}
			</div>
		{/if}

		<!-- Textarea -->
		<div class="flex-1 min-h-0 relative">
			<textarea
				bind:value={code}
				on:keydown={handleKeydown}
				class="w-full h-full resize-none bg-surface-900 text-surface-200 font-mono text-xs p-3 focus:outline-none placeholder-surface-600"
				placeholder="Enter code here..."
				spellcheck="false"
			/>
		</div>

		<!-- Execute button -->
		<div class="flex items-center gap-2 px-3 py-2 border-t border-surface-800 shrink-0">
			<button
				on:click={handleExecute}
				disabled={executing || !code.trim()}
				class="px-3 py-1.5 text-xs font-medium rounded transition-colors
					{executing || !code.trim()
						? 'bg-surface-800 text-surface-500 cursor-not-allowed'
						: 'bg-accent-600 text-[var(--oo-btn-primary-fg)] hover:bg-accent-500'}"
			>
				{#if executing}
					<span class="flex items-center gap-1.5">
						<div class="w-3 h-3 border-2 border-[var(--oo-fg-primary)]/30 border-t-[var(--oo-fg-primary)] rounded-full animate-spin" />
						Running...
					</span>
				{:else}
					Run
				{/if}
			</button>

			{#if result}
				<span class="text-xs text-surface-500">
					{result.execution_time.toFixed(1)}ms
					{#if result.return_code !== 0}
						<span class="text-[var(--oo-error)] ml-1">exit {result.return_code}</span>
					{/if}
				</span>
			{/if}
		</div>

		<!-- Output area -->
		{#if result}
			<div class="border-t border-surface-800 shrink-0 max-h-[40%] overflow-y-auto">
				<div class="px-3 py-1.5 text-xs text-surface-500 border-b border-surface-800/50 flex items-center gap-2">
					<span>Output</span>
					{#if result.success}
						<span class="text-[var(--oo-success)]">success</span>
					{:else}
						<span class="text-[var(--oo-error)]">error</span>
					{/if}
					{#if result.truncated}
						<span class="text-[var(--oo-warning)]">(truncated)</span>
					{/if}
				</div>

				{#if result.stdout}
					<pre class="text-xs font-mono text-surface-300 p-3 whitespace-pre-wrap break-words">{result.stdout}</pre>
				{/if}

				{#if result.stderr}
					<pre class="text-xs font-mono text-[var(--oo-error)]/80 p-3 whitespace-pre-wrap break-words border-t border-surface-800/30">{result.stderr}</pre>
				{/if}

				{#if result.error_message}
					<pre class="text-xs font-mono text-[var(--oo-error)] p-3 whitespace-pre-wrap break-words border-t border-surface-800/30">{result.error_message}</pre>
				{/if}

				{#if !result.stdout && !result.stderr && !result.error_message}
					<p class="text-xs text-surface-500 p-3 italic">No output</p>
				{/if}

				{#if result.output_files.length > 0}
					<div class="px-3 py-2 border-t border-surface-800/30">
						<p class="text-xs text-surface-500 mb-1">Output files:</p>
						{#each result.output_files as file}
							<span class="text-xs font-mono text-accent-400 block">{file}</span>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>
