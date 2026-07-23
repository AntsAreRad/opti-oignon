<!--
  BenchmarkProfiles.svelte
  The "Profiles" section extracted from BenchmarkV2Panel: list, create, edit
  and delete custom evaluation profiles with a question preview. Self-contained;
  behaviour unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { BenchmarkV2Profile, BenchmarkV2CustomProfile, BenchmarkV2QuestionPreview } from '$lib/types';
	import {
		getProfiles,
		getCustomProfiles,
		createCustomProfile,
		updateCustomProfile,
		deleteCustomProfile,
		previewProfileQuestions,
	} from '$lib/api/benchmarkV2';

	let profiles: BenchmarkV2Profile[] = [];
	let customProfiles: BenchmarkV2CustomProfile[] = [];
	let customProfilesLoading = false;
	let availableCategories: string[] = [];

	let editorMode: 'create' | 'edit' | null = null;
	let editorId = '';
	let editorName = '';
	let editorDescription = '';
	let editorCategories: string[] = [];
	let editorWeightPreset = 'balanced';
	let editorUseCustomWeights = false;
	let editorWeights = { accuracy: 0.35, code: 0.25, structure: 0.25, speed: 0.15 };
	let editorSaving = false;
	let editorError = '';
	let questionPreview: BenchmarkV2QuestionPreview | null = null;
	let previewLoading = false;
	let deleteConfirmId = '';

	onMount(async () => {
		await loadProfiles();
		await loadCustomProfiles();
	});

	async function loadProfiles() {
		try {
			const data = await getProfiles();
			profiles = data.profiles;
			availableCategories = data.available_categories || [];
		} catch {
			// silent
		}
	}

	async function loadCustomProfiles() {
		customProfilesLoading = true;
		try {
			const data = await getCustomProfiles();
			customProfiles = data.profiles;
		} catch {
			// silent
		} finally {
			customProfilesLoading = false;
		}
	}

	function openEditor(mode: 'create' | 'edit', profile?: BenchmarkV2CustomProfile) {
		editorMode = mode;
		editorError = '';
		questionPreview = null;
		if (mode === 'edit' && profile) {
			editorId = profile.profile_id;
			editorName = profile.name;
			editorDescription = profile.description;
			editorCategories = [...profile.categories];
			editorWeightPreset = profile.weight_preset;
			editorUseCustomWeights = !!profile.custom_weights;
			editorWeights = profile.custom_weights
				? { ...profile.custom_weights } as typeof editorWeights
				: { accuracy: 0.35, code: 0.25, structure: 0.25, speed: 0.15 };
		} else {
			editorId = '';
			editorName = '';
			editorDescription = '';
			editorCategories = [];
			editorWeightPreset = 'balanced';
			editorUseCustomWeights = false;
			editorWeights = { accuracy: 0.35, code: 0.25, structure: 0.25, speed: 0.15 };
		}
	}

	function closeEditor() {
		editorMode = null;
	}

	function toggleEditorCategory(cat: string) {
		if (editorCategories.includes(cat)) {
			editorCategories = editorCategories.filter((c) => c !== cat);
		} else {
			editorCategories = [...editorCategories, cat];
		}
	}

	async function loadPreview() {
		if (editorCategories.length === 0) {
			questionPreview = null;
			return;
		}
		previewLoading = true;
		try {
			questionPreview = await previewProfileQuestions(editorCategories);
		} catch {
			questionPreview = null;
		} finally {
			previewLoading = false;
		}
	}

	async function handleSaveProfile() {
		if (!editorName.trim()) {
			editorError = 'Name is required';
			return;
		}
		if (editorCategories.length === 0) {
			editorError = 'Select at least one category';
			return;
		}
		editorSaving = true;
		editorError = '';
		try {
			const payload = {
				name: editorName.trim(),
				description: editorDescription,
				categories: editorCategories,
				weight_preset: editorUseCustomWeights ? 'custom' : editorWeightPreset,
				custom_weights: editorUseCustomWeights ? editorWeights : undefined,
			};
			if (editorMode === 'create') {
				await createCustomProfile(payload);
			} else if (editorMode === 'edit' && editorId) {
				await updateCustomProfile(editorId, payload);
			}
			closeEditor();
			await loadCustomProfiles();
			await loadProfiles();
		} catch (e) {
			editorError = `Save failed: ${e}`;
		} finally {
			editorSaving = false;
		}
	}

	async function handleDeleteProfile(profileId: string) {
		try {
			await deleteCustomProfile(profileId);
			deleteConfirmId = '';
			await loadCustomProfiles();
			await loadProfiles();
		} catch {
			// silent
		}
	}
</script>

		<div class="bv2-section">
			<!-- Profile editor overlay -->
			{#if editorMode}
				<div class="bv2-editor">
					<h3 class="bv2-editor-title">
						{editorMode === 'create' ? 'New Custom Profile' : 'Edit Profile'}
					</h3>

					<div class="bv2-field">
						<label class="bv2-label" for="editor-name">Name</label>
						<input
							id="editor-name"
							type="text"
							class="bv2-input"
							bind:value={editorName}
							placeholder="My Custom Profile"
						/>
					</div>

					<div class="bv2-field">
						<label class="bv2-label" for="editor-desc">Description</label>
						<input
							id="editor-desc"
							type="text"
							class="bv2-input"
							bind:value={editorDescription}
							placeholder="What this profile evaluates"
						/>
					</div>

					<div class="bv2-field">
						<span class="bv2-label">Categories</span>
						<div class="bv2-category-grid">
							{#each availableCategories as cat}
								<label class="bv2-category-check">
									<input
										type="checkbox"
										checked={editorCategories.includes(cat)}
										on:change={() => toggleEditorCategory(cat)}
									/>
									<span>{cat.replace(/_/g, ' ')}</span>
								</label>
							{/each}
						</div>
						<button class="bv2-link-btn" on:click={loadPreview} disabled={previewLoading}>
							{previewLoading ? 'Loading...' : 'Preview questions'}
						</button>
						{#if questionPreview}
							<div class="bv2-preview-box">
								{#each Object.entries(questionPreview.category_counts) as [cat, count]}
									<span class="bv2-preview-item">{cat.replace(/_/g, ' ')}: {count}</span>
								{/each}
								<span class="bv2-preview-total">Total: {questionPreview.total} questions</span>
							</div>
						{/if}
					</div>

					<div class="bv2-field">
						<span class="bv2-label">Weights</span>
						<label class="bv2-toggle-row">
							<input type="checkbox" bind:checked={editorUseCustomWeights} />
							<span class="bv2-label-inline">Use custom weights</span>
						</label>
						{#if editorUseCustomWeights}
							<div class="bv2-weight-sliders">
								{#each Object.entries(editorWeights) as [key, val]}
									<div class="bv2-weight-row">
										<span class="bv2-weight-label">{key}</span>
										<input
											type="range"
											min="0"
											max="1"
											step="0.05"
											value={val}
											on:input={(e) => {
												editorWeights = { ...editorWeights, [key]: parseFloat(e.currentTarget.value) };
											}}
										/>
										<span class="bv2-weight-val">{val.toFixed(2)}</span>
									</div>
								{/each}
							</div>
						{:else}
							<select class="bv2-select bv2-select-sm" bind:value={editorWeightPreset}>
								<option value="balanced">Balanced</option>
								<option value="accuracy_first">Accuracy First</option>
								<option value="speed_first">Speed First</option>
								<option value="code_focused">Code Focused</option>
								<option value="writing_focused">Writing Focused</option>
							</select>
						{/if}
					</div>

					{#if editorError}
						<p class="bv2-error">{editorError}</p>
					{/if}

					<div class="bv2-editor-actions">
						<button class="bv2-btn bv2-btn-primary" on:click={handleSaveProfile} disabled={editorSaving}>
							{editorSaving ? 'Saving...' : 'Save Profile'}
						</button>
						<button class="bv2-btn bv2-btn-secondary" on:click={closeEditor}>Cancel</button>
					</div>
				</div>
			{:else}
				<!-- Profile list -->
				<div class="bv2-profiles-header">
					<button class="bv2-btn bv2-btn-primary" on:click={() => openEditor('create')}>
						New Profile
					</button>
				</div>

				{#if customProfilesLoading}
					<p class="bv2-hint">Loading custom profiles...</p>
				{:else if customProfiles.length === 0}
					<p class="bv2-hint">No custom profiles yet. Click "New Profile" to create one.</p>
				{:else}
					<div class="bv2-profiles-list">
						{#each customProfiles as cp}
							<div class="bv2-profile-card">
								<div class="bv2-profile-card-header">
									<span class="bv2-profile-card-name">{cp.name}</span>
									<span class="bv2-badge bv2-badge-custom">Custom</span>
								</div>
								{#if cp.description}
									<p class="bv2-profile-card-desc">{cp.description}</p>
								{/if}
								<div class="bv2-profile-card-cats">
									{#each cp.categories as cat}
										<span class="bv2-cat-tag">{cat.replace(/_/g, ' ')}</span>
									{/each}
								</div>
								<div class="bv2-profile-card-meta">
									<span>Weights: {cp.custom_weights ? 'Custom' : cp.weight_preset}</span>
								</div>
								<div class="bv2-profile-card-actions">
									<button class="bv2-link-btn" on:click={() => openEditor('edit', cp)}>Edit</button>
									{#if deleteConfirmId === cp.profile_id}
										<button class="bv2-link-btn bv2-link-danger" on:click={() => handleDeleteProfile(cp.profile_id)}>
											Confirm delete
										</button>
										<button class="bv2-link-btn" on:click={() => (deleteConfirmId = '')}>Cancel</button>
									{:else}
										<button class="bv2-link-btn bv2-link-danger" on:click={() => (deleteConfirmId = cp.profile_id)}>
											Delete
										</button>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				{/if}

				<!-- Built-in profiles reference -->
				{#if profiles.filter((p) => !p.custom).length > 0}
					<div class="bv2-builtin-section">
						<h4 class="bv2-subtitle">Built-in Profiles</h4>
						<div class="bv2-profiles-list">
							{#each profiles.filter((p) => !p.custom) as bp}
								<div class="bv2-profile-card bv2-profile-builtin">
									<div class="bv2-profile-card-header">
										<span class="bv2-profile-card-name">{bp.name}</span>
										<span class="bv2-badge">Built-in</span>
									</div>
									{#if bp.description}
										<p class="bv2-profile-card-desc">{bp.description}</p>
									{/if}
									<div class="bv2-profile-card-cats">
										{#each bp.categories as cat}
											<span class="bv2-cat-tag">{cat.replace(/_/g, ' ')}</span>
										{/each}
									</div>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			{/if}
		</div>
