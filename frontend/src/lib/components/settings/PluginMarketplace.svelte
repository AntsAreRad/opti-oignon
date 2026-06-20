<!--
  PluginMarketplace.svelte -- S102 Plugin Marketplace UI.

  Browse available plugins from the index, search by keyword/tag/author/hook,
  one-click install from URL, view and submit reviews.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Modal } from '$lib/ds';
	import {
		browseMarketplace,
		searchMarketplace,
		installFromUrl,
		getPluginReviews,
		addPluginReview,
		generatePluginTemplate,
	} from '$lib/api/pluginMarketplace';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		MarketplaceEntry,
		PluginReview,
		ReviewListResponse,
	} from '$lib/types';

	// ---- State: browse ----
	let entries: MarketplaceEntry[] = [];
	let total = 0;
	let loading = true;
	let error = '';
	let sortBy = 'stars';

	// ---- State: search ----
	let searchQuery = '';
	let searchTag = '';
	let searchAuthor = '';
	let searchHook = '';
	let isSearchMode = false;

	// ---- State: install from URL ----
	let showInstallUrl = false;
	let installUrl = '';
	let installHash = '';
	let installAutoEnable = true;
	let installing = false;

	// ---- State: detail/reviews panel ----
	let selectedEntry: MarketplaceEntry | null = null;
	let reviewsData: ReviewListResponse | null = null;
	let reviewsLoading = false;

	// ---- State: add review ----
	let newRating = 5;
	let newTitle = '';
	let newText = '';
	let submittingReview = false;

	// ---- State: template generator ----
	let showTemplateForm = false;
	let tmplName = '';
	let tmplAuthor = '';
	let tmplDesc = '';
	let tmplHooks = 'post_inference';
	let generatingTemplate = false;

	// ---- Load ----
	async function load() {
		loading = true;
		error = '';
		isSearchMode = false;
		try {
			const resp = await browseMarketplace({ sortBy, limit: 100, refresh: false });
			entries = resp.plugins;
			total = resp.total;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load marketplace';
		} finally {
			loading = false;
		}
	}

	async function handleRefresh() {
		loading = true;
		error = '';
		try {
			const resp = await browseMarketplace({ sortBy, limit: 100, refresh: true });
			entries = resp.plugins;
			total = resp.total;
			toastSuccess('Marketplace index refreshed');
		} catch (e) {
			error = e instanceof Error ? e.message : 'Refresh failed';
		} finally {
			loading = false;
		}
	}

	async function handleSearch() {
		if (!searchQuery && !searchTag && !searchAuthor && !searchHook) {
			await load();
			return;
		}
		loading = true;
		error = '';
		isSearchMode = true;
		try {
			const resp = await searchMarketplace({
				keyword: searchQuery,
				tag: searchTag,
				author: searchAuthor,
				hook: searchHook,
				sortBy,
				limit: 100,
			});
			entries = resp.plugins;
			total = resp.total;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Search failed';
		} finally {
			loading = false;
		}
	}

	function clearSearch() {
		searchQuery = '';
		searchTag = '';
		searchAuthor = '';
		searchHook = '';
		isSearchMode = false;
		load();
	}

	// ---- Install from URL ----
	async function handleInstallUrl() {
		if (!installUrl.trim()) return;
		installing = true;
		try {
			const resp = await installFromUrl(installUrl.trim(), installHash.trim(), installAutoEnable);
			if (resp.success) {
				toastSuccess(resp.message);
				installUrl = '';
				installHash = '';
				showInstallUrl = false;
			} else {
				toastError(resp.error || 'Installation failed');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Installation failed');
		} finally {
			installing = false;
		}
	}

	// ---- One-click install from index entry ----
	async function handleQuickInstall(entry: MarketplaceEntry) {
		if (!entry.url) {
			toastError('No download URL available for this plugin');
			return;
		}
		installing = true;
		try {
			const resp = await installFromUrl(entry.url, entry.sha256, true);
			if (resp.success) {
				toastSuccess(resp.message);
			} else {
				toastError(resp.error || 'Installation failed');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Installation failed');
		} finally {
			installing = false;
		}
	}

	// ---- Reviews ----
	async function openReviews(entry: MarketplaceEntry) {
		selectedEntry = entry;
		reviewsData = null;
		reviewsLoading = true;
		newRating = 5;
		newTitle = '';
		newText = '';
		try {
			reviewsData = await getPluginReviews(entry.name, { limit: 20 });
		} catch {
			// Reviews may not be available
		} finally {
			reviewsLoading = false;
		}
	}

	function closeReviews() {
		selectedEntry = null;
		reviewsData = null;
	}

	async function handleSubmitReview() {
		if (!selectedEntry) return;
		submittingReview = true;
		try {
			const resp = await addPluginReview(selectedEntry.name, newRating, {
				title: newTitle,
				text: newText,
			});
			if (resp.success) {
				toastSuccess('Review submitted');
				newTitle = '';
				newText = '';
				// Reload reviews
				reviewsData = await getPluginReviews(selectedEntry.name, { limit: 20 });
			} else {
				toastError(resp.error || 'Failed to submit review');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to submit review');
		} finally {
			submittingReview = false;
		}
	}

	// ---- Template ----
	async function handleGenerateTemplate() {
		if (!tmplName.trim()) return;
		generatingTemplate = true;
		try {
			const hooks = tmplHooks.split(',').map((h) => h.trim()).filter(Boolean);
			const resp = await generatePluginTemplate({
				name: tmplName.trim(),
				author: tmplAuthor.trim() || 'Your Name',
				description: tmplDesc.trim() || 'A custom Opti-Oignon plugin.',
				hooks,
			});
			if (resp.success) {
				toastSuccess(`Plugin scaffold created at ${resp.path}`);
				showTemplateForm = false;
				tmplName = '';
				tmplAuthor = '';
				tmplDesc = '';
				tmplHooks = 'post_inference';
			} else {
				toastError(resp.error || 'Template generation failed');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Template generation failed');
		} finally {
			generatingTemplate = false;
		}
	}

	// ---- Helpers ----
	function renderStars(rating: number): string {
		const full = Math.round(rating);
		return Array.from({ length: 5 }, (_, i) => (i < full ? '\u2605' : '\u2606')).join('');
	}

	function formatTimestamp(ts: number): string {
		if (!ts) return '';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
		});
	}

	onMount(load);
</script>

<div class="space-y-4">
	<!-- Header -->
	<div class="flex items-center justify-between flex-wrap gap-2">
		<div>
			<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">Marketplace</h2>
			<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
				Browse, search, and install community plugins.
				{#if !loading}
					{total} available.
				{/if}
			</p>
		</div>
		<div class="flex gap-2">
			<button
				on:click={() => { showTemplateForm = !showTemplateForm; showInstallUrl = false; }}
				class="px-3 py-1.5 rounded-lg text-xs font-medium"
				style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
			>
				{showTemplateForm ? 'Cancel' : 'New Plugin'}
			</button>
			<button
				on:click={() => { showInstallUrl = !showInstallUrl; showTemplateForm = false; }}
				class="px-3 py-1.5 rounded-lg text-xs font-medium"
				style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
			>
				{showInstallUrl ? 'Cancel' : 'Install from URL'}
			</button>
		</div>
	</div>

	<!-- Install from URL form -->
	{#if showInstallUrl}
		<div
			class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
		>
			<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">Install from URL</p>
			<div class="flex gap-2">
				<input
					type="text"
					bind:value={installUrl}
					placeholder="https://github.com/user/plugin-repo or .zip URL"
					class="flex-1 px-3 py-2 rounded-lg text-sm font-mono"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<button
					on:click={handleInstallUrl}
					disabled={installing || !installUrl.trim()}
					class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 shrink-0"
					style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
				>
					{installing ? 'Installing...' : 'Install'}
				</button>
			</div>
			<div class="flex items-center gap-4">
				<input
					type="text"
					bind:value={installHash}
					placeholder="SHA-256 hash (optional)"
					class="flex-1 px-3 py-1.5 rounded-lg text-xs font-mono"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<label class="flex items-center gap-2 text-xs shrink-0" style="color: var(--oo-fg-secondary);">
					<input type="checkbox" bind:checked={installAutoEnable} />
					Auto-enable
				</label>
			</div>
		</div>
	{/if}

	<!-- Template generator form -->
	{#if showTemplateForm}
		<div
			class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
		>
			<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">New Plugin Scaffold</p>
			<div class="grid grid-cols-2 gap-3">
				<input
					type="text"
					bind:value={tmplName}
					placeholder="plugin-name (lowercase)"
					class="px-3 py-2 rounded-lg text-sm font-mono"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<input
					type="text"
					bind:value={tmplAuthor}
					placeholder="Author"
					class="px-3 py-2 rounded-lg text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
			</div>
			<input
				type="text"
				bind:value={tmplDesc}
				placeholder="Short description"
				class="w-full px-3 py-2 rounded-lg text-sm"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			/>
			<div class="flex gap-2 items-end">
				<div class="flex-1">
					<label class="text-xs" style="color: var(--oo-fg-muted);">Hooks (comma-separated)</label>
					<input
						type="text"
						bind:value={tmplHooks}
						placeholder="post_inference, pre_prompt"
						class="w-full px-3 py-2 rounded-lg text-sm font-mono"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					/>
				</div>
				<button
					on:click={handleGenerateTemplate}
					disabled={generatingTemplate || !tmplName.trim()}
					class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 shrink-0"
					style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
				>
					{generatingTemplate ? 'Generating...' : 'Generate'}
				</button>
			</div>
		</div>
	{/if}

	<!-- Search bar -->
	<div class="flex gap-2 flex-wrap">
		<input
			type="text"
			bind:value={searchQuery}
			placeholder="Search plugins..."
			on:keydown={(e) => { if (e.key === 'Enter') handleSearch(); }}
			class="flex-1 min-w-48 px-3 py-2 rounded-lg text-sm"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
		/>
		<select
			bind:value={sortBy}
			on:change={isSearchMode ? handleSearch : load}
			class="px-3 py-2 rounded-lg text-sm"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
		>
			<option value="stars">Top rated</option>
			<option value="downloads">Most downloaded</option>
			<option value="updated_at">Recently updated</option>
			<option value="name">Alphabetical</option>
		</select>
		<button
			on:click={handleSearch}
			class="px-3 py-2 rounded-lg text-sm font-medium"
			style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
		>
			Search
		</button>
		{#if isSearchMode}
			<button
				on:click={clearSearch}
				class="px-3 py-2 rounded-lg text-sm"
				style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
			>
				Clear
			</button>
		{/if}
		<button
			on:click={handleRefresh}
			class="px-3 py-2 rounded-lg text-sm"
			title="Refresh index from remote"
			style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
		>
			Refresh
		</button>
	</div>

	<!-- Loading / Error / Empty -->
	{#if loading}
		<div class="py-8 text-center text-sm" style="color: var(--oo-fg-muted);">
			Loading marketplace...
		</div>
	{:else if error}
		<div
			class="py-4 px-4 rounded-lg text-sm"
			style="background-color: var(--oo-error-bg); color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
		>
			{error}
		</div>
	{:else if entries.length === 0}
		<div class="py-8 text-center text-sm" style="color: var(--oo-fg-muted);">
			{#if isSearchMode}
				No plugins match your search.
			{:else}
				No plugins in the index. Configure a remote index URL or add plugins manually.
			{/if}
		</div>
	{:else}
		<!-- Plugin grid -->
		<div class="grid gap-3" style="grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));">
			{#each entries as entry (entry.name)}
				<div
					class="rounded-lg overflow-hidden flex flex-col"
					style="border: 1px solid var(--oo-bd-subtle); background-color: var(--oo-bg-elevated);"
				>
					<!-- Card header -->
					<div class="px-4 pt-3 pb-2 flex-1">
						<div class="flex items-start justify-between">
							<div class="min-w-0 flex-1">
								<div class="flex items-center gap-2">
									<span class="text-sm font-medium truncate" style="color: var(--oo-fg-primary);">
										{entry.name}
									</span>
									<span class="text-xs font-mono" style="color: var(--oo-fg-muted);">
										v{entry.version}
									</span>
								</div>
								<p class="text-xs mt-0.5" style="color: var(--oo-fg-secondary);">
									by {entry.author}
								</p>
							</div>
							<!-- Stars display -->
							{#if entry.average_rating > 0}
								<div class="text-right shrink-0 ml-2">
									<span class="text-xs" style="color: var(--oo-acc-600);">
										{renderStars(entry.average_rating)}
									</span>
									<p class="text-xs" style="color: var(--oo-fg-muted);">
										{entry.average_rating.toFixed(1)} ({entry.review_count})
									</p>
								</div>
							{/if}
						</div>
						<p class="text-xs mt-2 line-clamp-2" style="color: var(--oo-fg-secondary);">
							{entry.description}
						</p>
						<!-- Tags -->
						{#if entry.tags.length > 0}
							<div class="flex flex-wrap gap-1 mt-2">
								{#each entry.tags.slice(0, 4) as tag}
									<span
										class="text-xs px-1.5 py-0.5 rounded"
										style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);"
									>
										{tag}
									</span>
								{/each}
								{#if entry.tags.length > 4}
									<span class="text-xs" style="color: var(--oo-fg-muted);">+{entry.tags.length - 4}</span>
								{/if}
							</div>
						{/if}
						<!-- Stats row -->
						<div class="flex items-center gap-4 mt-2 text-xs" style="color: var(--oo-fg-muted);">
							{#if entry.downloads > 0}
								<span>{entry.downloads} downloads</span>
							{/if}
							{#if entry.hooks.length > 0}
								<span>{entry.hooks.length} hooks</span>
							{/if}
						</div>
					</div>
					<!-- Card actions -->
					<div
						class="px-4 py-2 flex items-center justify-between"
						style="border-top: 1px solid var(--oo-bd-subtle);"
					>
						<button
							on:click={() => openReviews(entry)}
							class="text-xs px-2 py-1 rounded"
							style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
						>
							Reviews
						</button>
						<button
							on:click={() => handleQuickInstall(entry)}
							disabled={installing || !entry.url}
							class="text-xs px-3 py-1 rounded-lg font-medium disabled:opacity-50"
							style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
						>
							Install
						</button>
					</div>
				</div>
			{/each}
		</div>
	{/if}

	<!-- Review panel modal -->
	<Modal
		open={!!selectedEntry}
		variant="center"
		size="lg"
		title={selectedEntry?.name ?? 'Plugin details'}
		onClose={closeReviews}
	>
		{#if selectedEntry}
			<p class="text-xs" style="color: var(--oo-fg-muted); margin: 0 0 8px;">
				v{selectedEntry.version} by {selectedEntry.author}
			</p>

			<!-- Rating summary -->
				{#if reviewsData}
					<div class="flex items-center gap-3">
						<span class="text-lg" style="color: var(--oo-acc-600);">
							{renderStars(reviewsData.average_rating)}
						</span>
						<span class="text-sm" style="color: var(--oo-fg-primary);">
							{reviewsData.average_rating.toFixed(1)}
						</span>
						<span class="text-xs" style="color: var(--oo-fg-muted);">
							({reviewsData.total} reviews)
						</span>
					</div>

					<!-- Distribution bars -->
					<div class="space-y-1">
						{#each [5, 4, 3, 2, 1] as star}
							<div class="flex items-center gap-2 text-xs">
								<span class="w-3 text-right" style="color: var(--oo-fg-muted);">{star}</span>
								<div
									class="flex-1 h-2 rounded-full overflow-hidden"
									style="background-color: var(--oo-bg-overlay);"
								>
									<div
										class="h-full rounded-full"
										style="background-color: var(--oo-acc-600);
											width: {reviewsData.total > 0
												? ((reviewsData.rating_distribution[star] || 0) / reviewsData.total) * 100
												: 0}%;"
									></div>
								</div>
								<span class="w-6 text-right" style="color: var(--oo-fg-muted);">
									{reviewsData.rating_distribution[star] || 0}
								</span>
							</div>
						{/each}
					</div>

					<!-- Existing reviews -->
					{#if reviewsData.reviews.length > 0}
						<div class="space-y-3 pt-2" style="border-top: 1px solid var(--oo-bd-subtle);">
							{#each reviewsData.reviews as review (review.id)}
								<div class="space-y-1">
									<div class="flex items-center justify-between">
										<div class="flex items-center gap-2">
											<span class="text-xs" style="color: var(--oo-acc-600);">
												{renderStars(review.rating)}
											</span>
											{#if review.title}
												<span class="text-xs font-medium" style="color: var(--oo-fg-primary);">
													{review.title}
												</span>
											{/if}
										</div>
										<span class="text-xs" style="color: var(--oo-fg-muted);">
											{formatTimestamp(review.created_at)}
										</span>
									</div>
									{#if review.text}
										<p class="text-xs" style="color: var(--oo-fg-secondary);">
											{review.text}
										</p>
									{/if}
									<p class="text-xs" style="color: var(--oo-fg-muted);">
										-- {review.author}
									</p>
								</div>
							{/each}
						</div>
					{/if}
				{:else if reviewsLoading}
					<p class="text-xs" style="color: var(--oo-fg-muted);">Loading reviews...</p>
				{:else}
					<p class="text-xs" style="color: var(--oo-fg-muted);">No reviews yet.</p>
				{/if}

				<!-- Add review form -->
				<div class="pt-3 space-y-2" style="border-top: 1px solid var(--oo-bd-subtle);">
					<p class="text-xs font-medium" style="color: var(--oo-fg-primary);">Write a review</p>
					<div class="flex items-center gap-3">
						<label class="text-xs" style="color: var(--oo-fg-muted);">Rating:</label>
						<div class="flex gap-1">
							{#each [1, 2, 3, 4, 5] as star}
								<button
									on:click={() => { newRating = star; }}
									class="text-base"
									style="color: {star <= newRating ? 'var(--oo-acc-600)' : 'var(--oo-fg-muted)'};"
								>
									{star <= newRating ? '\u2605' : '\u2606'}
								</button>
							{/each}
						</div>
					</div>
					<input
						type="text"
						bind:value={newTitle}
						placeholder="Review title (optional)"
						class="w-full px-3 py-1.5 rounded-lg text-xs"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					/>
					<textarea
						bind:value={newText}
						placeholder="Your review..."
						rows="3"
						class="w-full px-3 py-1.5 rounded-lg text-xs"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd);
							color: var(--oo-fg-primary); resize: vertical;"
					></textarea>
					<div class="flex gap-2">
						<button
							on:click={handleSubmitReview}
							disabled={submittingReview}
							class="px-4 py-1.5 rounded-lg text-xs font-medium disabled:opacity-50"
							style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
						>
							{submittingReview ? 'Submitting...' : 'Submit'}
						</button>
					</div>
				</div>
		{/if}
	</Modal>
</div>
