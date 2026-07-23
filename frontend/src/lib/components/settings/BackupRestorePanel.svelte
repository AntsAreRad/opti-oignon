<!--
  BackupRestorePanel.svelte -- Backup/Restore UI.

  Export section: checkboxes per section, "Export All" button, download trigger.
  Import section: file upload zone, preview display, merge/replace strategy toggle.
  Preview diff: table showing what would be added/changed/skipped.
  Import confirmation with summary.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		listBackupSections,
		downloadBackup,
		previewImport,
		importBackup,
	} from '$lib/api/backup';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		BackupSectionInfo,
		BackupData,
		BackupPreviewResponse,
		BackupImportResponse,
		BackupDiffItem,
	} from '$lib/types';

	// -- Export state --
	let sections: BackupSectionInfo[] = [];
	let selectedSections: Set<string> = new Set();
	let loadingSections = true;
	let exporting = false;

	// -- Import state --
	let importFile: File | null = null;
	let importData: BackupData | null = null;
	let strategy: 'merge' | 'replace' = 'merge';
	let preview: BackupPreviewResponse | null = null;
	let previewing = false;
	let importing = false;
	let importResult: BackupImportResponse | null = null;
	let parseError = '';
	let dragOver = false;

	onMount(async () => {
		await loadSections();
	});

	async function loadSections() {
		loadingSections = true;
		try {
			const resp = await listBackupSections();
			sections = resp.sections;
			// Select all available sections by default
			selectedSections = new Set(
				sections.filter((s) => s.available).map((s) => s.name)
			);
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			toastError('Failed to load backup sections: ' + msg);
		} finally {
			loadingSections = false;
		}
	}

	function toggleSection(name: string) {
		if (selectedSections.has(name)) {
			selectedSections.delete(name);
		} else {
			selectedSections.add(name);
		}
		selectedSections = selectedSections; // trigger reactivity
	}

	function toggleAllSections() {
		const available = sections.filter((s) => s.available).map((s) => s.name);
		if (selectedSections.size === available.length) {
			selectedSections = new Set();
		} else {
			selectedSections = new Set(available);
		}
	}

	async function handleExport() {
		exporting = true;
		try {
			const sectionList = selectedSections.size === sections.filter((s) => s.available).length
				? undefined
				: [...selectedSections].join(',');
			await downloadBackup(sectionList);
			toastSuccess('Backup exported successfully');
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			toastError('Export failed: ' + msg);
		} finally {
			exporting = false;
		}
	}

	function handleFileDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const files = e.dataTransfer?.files;
		if (files && files.length > 0) {
			processFile(files[0]);
		}
	}

	function handleFileSelect(e: Event) {
		const target = e.target as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			processFile(target.files[0]);
		}
	}

	async function processFile(file: File) {
		importFile = file;
		importData = null;
		preview = null;
		importResult = null;
		parseError = '';

		try {
			const text = await file.text();
			const data = JSON.parse(text);
			if (!data.schema_version || !data.sections) {
				parseError = 'Invalid backup file: missing schema_version or sections.';
				return;
			}
			importData = data as BackupData;
		} catch {
			parseError = 'Failed to parse JSON file. Is this a valid .oo-backup.json?';
		}
	}

	async function handlePreview() {
		if (!importData) return;
		previewing = true;
		preview = null;
		importResult = null;
		try {
			preview = await previewImport(importData, strategy);
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			toastError('Preview failed: ' + msg);
		} finally {
			previewing = false;
		}
	}

	async function handleImport() {
		if (!importData) return;
		importing = true;
		importResult = null;
		try {
			importResult = await importBackup(importData, strategy);
			if (importResult.success) {
				toastSuccess(
					`Backup imported: ${importResult.sections_imported.length} sections applied`
				);
			} else {
				toastError(
					`Import partially failed: ${importResult.errors.length} error(s)`
				);
			}
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			toastError('Import failed: ' + msg);
		} finally {
			importing = false;
		}
	}

	function resetImport() {
		importFile = null;
		importData = null;
		preview = null;
		importResult = null;
		parseError = '';
	}

	function actionLabel(action: string): string {
		switch (action) {
			case 'add': return 'Add';
			case 'update': return 'Update';
			case 'skip': return 'Skip';
			default: return action;
		}
	}

	function actionColor(action: string): string {
		switch (action) {
			case 'add': return 'var(--oo-success)';
			case 'update': return 'var(--oo-warning)';
			case 'skip': return 'var(--oo-fg-muted)';
			default: return 'var(--oo-fg-secondary)';
		}
	}
</script>

<div class="space-y-6">
	<!-- Header -->
	<div>
		<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">
			Backup & Restore
		</h2>
		<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
			Export your configuration to a file or restore from a previous backup.
		</p>
	</div>

	<!-- ==================== EXPORT ==================== -->
	<div class="rounded-lg p-4" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
		<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
			Export Configuration
		</h3>

		{#if loadingSections}
			<p class="text-xs" style="color: var(--oo-fg-muted);">Loading sections...</p>
		{:else}
			<!-- Select all toggle -->
			<div class="flex items-center gap-2 mb-3">
				<button
					on:click={toggleAllSections}
					class="text-xs px-2 py-1 rounded"
					style="color: var(--oo-acc-400); border: 1px solid var(--oo-bd-subtle);"
				>
					{selectedSections.size === sections.filter((s) => s.available).length
						? 'Deselect All'
						: 'Select All'}
				</button>
				<span class="text-xs" style="color: var(--oo-fg-muted);">
					{selectedSections.size} of {sections.filter((s) => s.available).length} sections selected
				</span>
			</div>

			<!-- Section checkboxes grid -->
			<div class="grid grid-cols-2 gap-2 mb-4">
				{#each sections as section}
					<label
						class="flex items-start gap-2 p-2 rounded cursor-pointer text-xs transition-colors"
						style="border: 1px solid var(--oo-bd-subtle); opacity: {section.available ? 1 : 0.4};"
					>
						<input
							type="checkbox"
							checked={selectedSections.has(section.name)}
							disabled={!section.available}
							on:change={() => toggleSection(section.name)}
							class="mt-0.5"
						/>
						<div>
							<span class="font-medium" style="color: var(--oo-fg-primary);">{section.name}</span>
							<span class="ml-1" style="color: var(--oo-fg-muted);">({section.item_count})</span>
							<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">{section.description}</p>
						</div>
					</label>
				{/each}
			</div>

			<!-- Export button -->
			<button
				on:click={handleExport}
				disabled={exporting || selectedSections.size === 0}
				class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
				style="background-color: var(--oo-acc-500); color: var(--oo-fg-on-accent);"
			>
				{#if exporting}
					Exporting...
				{:else}
					Export {selectedSections.size === sections.filter((s) => s.available).length
						? 'All'
						: `${selectedSections.size} Sections`}
				{/if}
			</button>
		{/if}
	</div>

	<!-- ==================== IMPORT ==================== -->
	<div class="rounded-lg p-4" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
		<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
			Import Configuration
		</h3>

		{#if !importData && !importResult}
			<!-- Drop zone -->
			<div
				class="border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer"
				style="border-color: {dragOver
					? 'var(--oo-acc-400)'
					: 'var(--oo-bd-subtle)'}; background-color: {dragOver
					? 'var(--oo-bg-hover)'
					: 'transparent'};"
				on:dragover|preventDefault={() => (dragOver = true)}
				on:dragleave={() => (dragOver = false)}
				on:drop={handleFileDrop}
				on:click={() => document.getElementById('backup-file-input')?.click()}
				on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('backup-file-input')?.click(); }}
				role="button"
				tabindex="0"
			>
				<svg class="w-8 h-8 mx-auto mb-2" style="color: var(--oo-fg-muted);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
					<path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
				</svg>
				<p class="text-sm" style="color: var(--oo-fg-secondary);">
					Drop a <span class="font-mono">.oo-backup.json</span> file here or click to browse
				</p>
			</div>
			<input
				id="backup-file-input"
				type="file"
				accept=".json,.oo-backup.json"
				class="hidden"
				on:change={handleFileSelect}
			/>
			{#if parseError}
				<p class="text-xs mt-2" style="color: var(--oo-danger);">{parseError}</p>
			{/if}
		{:else if importData && !importResult}
			<!-- File loaded: show metadata and controls -->
			<div class="space-y-4">
				<!-- File info -->
				<div class="flex items-center justify-between p-3 rounded-lg"
					style="background-color: var(--oo-bg-default); border: 1px solid var(--oo-bd-subtle);">
					<div>
						<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">
							{importFile?.name ?? 'Backup file'}
						</p>
						<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
							Version {importData.metadata.opti_oignon_version}
							&middot; {importData.metadata.timestamp_iso}
							&middot; {importData.metadata.sections_included.length} sections
						</p>
					</div>
					<button
						on:click={resetImport}
						class="text-xs px-2 py-1 rounded"
						style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
					>
						Clear
					</button>
				</div>

				<!-- Strategy toggle -->
				<div>
					<label class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Import Strategy</label>
					<div class="flex gap-2 mt-1">
						<button
							on:click={() => { strategy = 'merge'; preview = null; }}
							class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
							style="{strategy === 'merge'
								? 'background-color: var(--oo-acc-500); color: var(--oo-fg-on-accent);'
								: 'background-color: var(--oo-bg-default); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);'}"
						>
							Merge
						</button>
						<button
							on:click={() => { strategy = 'replace'; preview = null; }}
							class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
							style="{strategy === 'replace'
								? 'background-color: var(--oo-warning); color: var(--oo-fg-on-accent);'
								: 'background-color: var(--oo-bg-default); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);'}"
						>
							Replace
						</button>
					</div>
					<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">
						{#if strategy === 'merge'}
							Keep existing values, add only missing ones from backup.
						{:else}
							Overwrite existing values with backup data.
						{/if}
					</p>
				</div>

				<!-- Preview button -->
				<button
					on:click={handlePreview}
					disabled={previewing}
					class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
					style="background-color: var(--oo-bg-default); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
				>
					{previewing ? 'Loading Preview...' : 'Preview Changes'}
				</button>

				<!-- Preview diff -->
				{#if preview}
					<div class="space-y-3">
						{#if preview.errors.length > 0}
							<div class="p-3 rounded-lg" style="background-color: var(--oo-danger-bg); border: 1px solid var(--oo-danger-bd);">
								<p class="text-xs font-medium" style="color: var(--oo-danger);">Validation Errors</p>
								{#each preview.errors as err}
									<p class="text-xs mt-1" style="color: var(--oo-danger);">{err}</p>
								{/each}
							</div>
						{/if}

						<!-- Summary badges -->
						<div class="flex gap-3">
							{#if preview.summary.add}
								<span class="text-xs px-2 py-0.5 rounded-full" style="background-color: var(--oo-success-bg); color: var(--oo-success);">
									+{preview.summary.add} add
								</span>
							{/if}
							{#if preview.summary.update}
								<span class="text-xs px-2 py-0.5 rounded-full" style="background-color: var(--oo-warning-bg); color: var(--oo-warning);">
									~{preview.summary.update} update
								</span>
							{/if}
							{#if preview.summary.skip}
								<span class="text-xs px-2 py-0.5 rounded-full" style="background-color: var(--oo-bg-default); color: var(--oo-fg-muted);">
									{preview.summary.skip} skip
								</span>
							{/if}
						</div>

						<!-- Diff table -->
						{#if preview.diff.length > 0}
							<div class="overflow-x-auto rounded-lg" style="border: 1px solid var(--oo-bd-subtle);">
								<table class="w-full text-xs">
									<thead>
										<tr style="background-color: var(--oo-bg-default); border-bottom: 1px solid var(--oo-bd-subtle);">
											<th class="px-3 py-2 text-left font-medium" style="color: var(--oo-fg-secondary);">Section</th>
											<th class="px-3 py-2 text-left font-medium" style="color: var(--oo-fg-secondary);">Key</th>
											<th class="px-3 py-2 text-left font-medium" style="color: var(--oo-fg-secondary);">Action</th>
										</tr>
									</thead>
									<tbody>
										{#each preview.diff as item}
											<tr style="border-bottom: 1px solid var(--oo-bd-subtle);">
												<td class="px-3 py-2 font-mono" style="color: var(--oo-fg-primary);">{item.section}</td>
												<td class="px-3 py-2 font-mono" style="color: var(--oo-fg-secondary);">{item.key}</td>
												<td class="px-3 py-2 font-medium" style="color: {actionColor(item.action)};">{actionLabel(item.action)}</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{:else if preview.valid && preview.errors.length === 0}
							<p class="text-xs" style="color: var(--oo-fg-muted);">No changes detected. Your configuration matches the backup.</p>
						{/if}

						<!-- Apply button -->
						{#if preview.valid && preview.diff.length > 0}
							<div class="flex items-center gap-3">
								<button
									on:click={handleImport}
									disabled={importing}
									class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
									style="background-color: var(--oo-acc-500); color: var(--oo-fg-on-accent);"
								>
									{importing ? 'Importing...' : `Apply Import (${strategy})`}
								</button>
								{#if strategy === 'replace'}
									<span class="text-xs" style="color: var(--oo-warning);">
										This will overwrite existing configuration.
									</span>
								{/if}
							</div>
						{/if}
					</div>
				{/if}
			</div>
		{:else if importResult}
			<!-- Import result -->
			<div class="space-y-3">
				<div class="p-3 rounded-lg"
					style="background-color: {importResult.success
						? 'var(--oo-success-bg)'
						: 'var(--oo-danger-bg)'}; border: 1px solid {importResult.success
						? 'var(--oo-success-bd)'
						: 'var(--oo-danger-bd)'};">
					<p class="text-sm font-medium" style="color: {importResult.success ? 'var(--oo-success)' : 'var(--oo-danger)'};">
						{importResult.success ? 'Import Successful' : 'Import Failed'}
					</p>
					{#if importResult.sections_imported.length > 0}
						<p class="text-xs mt-1" style="color: var(--oo-fg-secondary);">
							Imported: {importResult.sections_imported.join(', ')}
						</p>
					{/if}
					{#if importResult.sections_failed.length > 0}
						<p class="text-xs mt-1" style="color: var(--oo-danger);">
							Failed: {importResult.sections_failed.join(', ')}
						</p>
					{/if}
					{#if importResult.rolled_back}
						<p class="text-xs mt-1" style="color: var(--oo-warning);">
							Changes were rolled back due to errors.
						</p>
					{/if}
					{#each importResult.errors as err}
						<p class="text-xs mt-1" style="color: var(--oo-danger);">{err}</p>
					{/each}
				</div>
				<button
					on:click={resetImport}
					class="px-4 py-2 rounded-lg text-sm font-medium"
					style="background-color: var(--oo-bg-default); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
				>
					Start Over
				</button>
			</div>
		{/if}
	</div>
</div>
