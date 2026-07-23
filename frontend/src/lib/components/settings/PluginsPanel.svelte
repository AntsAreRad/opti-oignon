<!--
  PluginsPanel.svelte -- Plugin Management UI.

  Shows installed plugins with enable/disable toggles, plugin details
  (version, author, hooks, permissions), install from directory,
  plugin config editor, uninstall, and marketplace sub-tab.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import PluginMarketplace from './PluginMarketplace.svelte';
	import {
		listPlugins,
		enablePlugin,
		disablePlugin,
		uninstallPlugin,
		installPlugin,
		getPluginConfig,
		updatePluginConfig,
	} from '$lib/api/plugins';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		PluginInfo,
		PluginConfigResponse,
	} from '$lib/types';

	// State
	let plugins: PluginInfo[] = [];
	let loading = true;
	let error = '';

	// Install form
	let showInstallForm = false;
	let installDir = '';
	let installAutoEnable = true;
	let installing = false;

	// Detail / config panel
	let selectedPlugin: PluginInfo | null = null;
	let pluginConfig: PluginConfigResponse | null = null;
	let configLoading = false;
	let configEditorText = '';
	let savingConfig = false;

	// Confirm uninstall
	let confirmUninstall: string | null = null;

	// Sub-tab: installed vs marketplace
	type SubTab = 'installed' | 'marketplace';
	let subTab: SubTab = 'installed';

	// Known slash commands per plugin
	const SLASH_COMMANDS: Record<string, string[]> = {
		'scratchpad': ['/note <text>', '/notes', '/note delete <id>', '/note search <query>'],
		'task-extractor': ['/tasks', '/tasks done <id>', '/tasks clear'],
		'github-connector': ['/gh issues', '/gh pr', '/gh repo <owner/repo>', '/gh gist'],
		'session-summarizer': ['/summary'],
	};

	function usageHint(plugin: PluginInfo): { type: string; detail: string } | null {
		const cmds = SLASH_COMMANDS[plugin.name];
		if (cmds) return { type: 'commands', detail: cmds.join('  ') };
		const hasToolCall = plugin.hooks.includes('tool_call');
		const hasPostInference = plugin.hooks.includes('post_inference');
		if (hasPostInference && !hasToolCall) return { type: 'auto', detail: 'Runs automatically on every LLM response' };
		if (hasPostInference && hasToolCall) return { type: 'hybrid', detail: 'Runs automatically and responds to commands' };
		if (hasToolCall) return { type: 'commands', detail: 'Responds to slash commands' };
		return null;
	}

	async function load() {
		loading = true;
		error = '';
		try {
			const resp = await listPlugins();
			plugins = resp.plugins;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load plugins';
		} finally {
			loading = false;
		}
	}

	async function handleToggle(plugin: PluginInfo) {
		try {
			if (plugin.state === 'enabled') {
				const resp = await disablePlugin(plugin.name);
				if (resp.success) {
					toastSuccess(`Plugin "${plugin.name}" disabled`);
				} else {
					toastError(resp.error || 'Failed to disable');
				}
			} else {
				const resp = await enablePlugin(plugin.name);
				if (resp.success) {
					toastSuccess(`Plugin "${plugin.name}" enabled`);
				} else {
					toastError(resp.error || 'Failed to enable');
				}
			}
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Operation failed');
		}
	}

	async function handleInstall() {
		if (!installDir.trim()) return;
		installing = true;
		try {
			const resp = await installPlugin(installDir.trim(), installAutoEnable);
			if (resp.success) {
				toastSuccess(resp.message);
				installDir = '';
				showInstallForm = false;
				await load();
			} else {
				toastError(resp.error || 'Installation failed');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Installation failed');
		} finally {
			installing = false;
		}
	}

	async function handleUninstall(name: string) {
		try {
			const resp = await uninstallPlugin(name);
			if (resp.success) {
				toastSuccess(`Plugin "${name}" uninstalled`);
				if (selectedPlugin?.name === name) {
					selectedPlugin = null;
					pluginConfig = null;
				}
				confirmUninstall = null;
				await load();
			} else {
				toastError(resp.error || 'Uninstall failed');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Uninstall failed');
		}
	}

	async function openDetails(plugin: PluginInfo) {
		selectedPlugin = plugin;
		pluginConfig = null;
		configLoading = true;
		try {
			pluginConfig = await getPluginConfig(plugin.name);
			configEditorText = JSON.stringify(pluginConfig.config, null, 2);
		} catch {
			// Config may not be available
		} finally {
			configLoading = false;
		}
	}

	function closeDetails() {
		selectedPlugin = null;
		pluginConfig = null;
	}

	async function handleSaveConfig() {
		if (!selectedPlugin || !pluginConfig) return;
		savingConfig = true;
		try {
			const parsed = JSON.parse(configEditorText);
			const resp = await updatePluginConfig(selectedPlugin.name, parsed);
			if (resp.success) {
				toastSuccess(`Configuration updated for "${selectedPlugin.name}"`);
				pluginConfig = { ...pluginConfig, config: parsed };
			} else {
				toastError(resp.error || 'Failed to save config');
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Invalid JSON');
		} finally {
			savingConfig = false;
		}
	}

	function stateColor(state: string): string {
		switch (state) {
			case 'enabled': return 'var(--oo-success)';
			case 'disabled': return 'var(--oo-fg-muted)';
			default: return 'var(--oo-fg-secondary)';
		}
	}

	function formatTimestamp(ts: number): string {
		if (!ts) return 'N/A';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
		});
	}

	onMount(load);
</script>

<div class="space-y-4">
	<!-- Sub-tabs: Installed / Marketplace -->
	<div class="flex gap-1 rounded-lg p-0.5" style="background-color: var(--oo-bg-overlay);">
		<button
			on:click={() => { subTab = 'installed'; }}
			class="flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
			style="{subTab === 'installed'
				? 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary);'
				: 'color: var(--oo-fg-muted);'}"
		>
			Installed
		</button>
		<button
			on:click={() => { subTab = 'marketplace'; }}
			class="flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
			style="{subTab === 'marketplace'
				? 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary);'
				: 'color: var(--oo-fg-muted);'}"
		>
			Marketplace
		</button>
	</div>

	{#if subTab === 'marketplace'}
		<PluginMarketplace />
	{:else}
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">Plugins</h2>
			<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
				Manage extensions for tools, pipeline steps, and inference hooks.
				{#if !loading}
					{plugins.length} installed, {plugins.filter(p => p.state === 'enabled').length} enabled.
				{/if}
			</p>
		</div>
		<button
			on:click={() => { showInstallForm = !showInstallForm; }}
			class="px-3 py-1.5 rounded-lg text-sm font-medium"
			style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
		>
			{showInstallForm ? 'Cancel' : 'Install Plugin'}
		</button>
	</div>

	<!-- Install form -->
	{#if showInstallForm}
		<div class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
			<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">Install from directory</p>
			<div class="flex gap-2">
				<input
					type="text"
					bind:value={installDir}
					placeholder="/path/to/plugin/directory"
					class="flex-1 px-3 py-2 rounded-lg text-sm font-mono"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<button
					on:click={handleInstall}
					disabled={installing || !installDir.trim()}
					class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 shrink-0"
					style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
				>
					{installing ? 'Installing...' : 'Install'}
				</button>
			</div>
			<label class="flex items-center gap-2 text-xs" style="color: var(--oo-fg-secondary);">
				<input type="checkbox" bind:checked={installAutoEnable} />
				Auto-enable after install
			</label>
		</div>
	{/if}

	<!-- Loading / Error -->
	{#if loading}
		<div class="py-8 text-center text-sm" style="color: var(--oo-fg-muted);">
			Loading plugins...
		</div>
	{:else if error}
		<div class="py-4 px-4 rounded-lg text-sm"
			style="background-color: var(--oo-error-bg); color: var(--oo-error); border: 1px solid var(--oo-error-bd);">
			{error}
		</div>
	{:else if plugins.length === 0}
		<div class="py-8 text-center text-sm" style="color: var(--oo-fg-muted);">
			No plugins installed. Click "Install Plugin" to add one.
		</div>
	{:else}
		<!-- Plugin list -->
		<div class="space-y-2">
			{#each plugins as plugin (plugin.name)}
				<div
					class="rounded-lg overflow-hidden transition-colors"
					style="border: 1px solid var(--oo-bd-subtle); background-color: var(--oo-bg-elevated);"
				>
					<div class="flex items-center justify-between px-4 py-3">
						<!-- Left: info -->
						<button
							class="flex-1 text-left flex items-center gap-3"
							on:click={() => openDetails(plugin)}
						>
							<!-- State dot -->
							<span
								class="w-2.5 h-2.5 rounded-full shrink-0"
								style="background-color: {stateColor(plugin.state)};"
							></span>
							<div class="min-w-0">
								<div class="flex items-center gap-2">
									<span class="text-sm font-medium truncate" style="color: var(--oo-fg-primary);">
										{plugin.name}
									</span>
									<span class="text-xs font-mono" style="color: var(--oo-fg-muted);">
										v{plugin.version}
									</span>
								</div>
								<p class="text-xs truncate mt-0.5" style="color: var(--oo-fg-secondary);">
									{plugin.description}
								</p>
								{#if usageHint(plugin)}
									<p class="text-[10px] font-mono mt-0.5 truncate" style="color: var(--oo-fg-tertiary);">
										{usageHint(plugin)?.detail}
									</p>
								{/if}
							</div>
						</button>

						<!-- Right: toggle + actions -->
						<div class="flex items-center gap-2 shrink-0 ml-3">
							<!-- Hooks badges -->
							{#each plugin.hooks.slice(0, 3) as hook}
								<span class="text-xs px-1.5 py-0.5 rounded"
									style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);">
									{hook}
								</span>
							{/each}
							{#if plugin.hooks.length > 3}
								<span class="text-xs" style="color: var(--oo-fg-muted);">
									+{plugin.hooks.length - 3}
								</span>
							{/if}

							<!-- Toggle -->
							<button
								on:click|stopPropagation={() => handleToggle(plugin)}
								class="relative w-10 h-5 rounded-full transition-colors"
								style="background-color: {plugin.state === 'enabled'
									? 'var(--oo-acc-600)'
									: 'var(--oo-bg-overlay)'};"
								title="{plugin.state === 'enabled' ? 'Disable' : 'Enable'} plugin"
							>
								<span
									class="absolute top-0.5 w-4 h-4 rounded-full transition-transform"
									style="background-color: var(--oo-fg-on-accent);
										transform: translateX({plugin.state === 'enabled' ? '22px' : '2px'});"
								></span>
							</button>
						</div>
					</div>

					<!-- Detail panel (expanded) -->
					{#if selectedPlugin?.name === plugin.name}
						<div class="px-4 pb-4 space-y-3" style="border-top: 1px solid var(--oo-bd-subtle);">
							<!-- Metadata -->
							<div class="grid grid-cols-2 gap-x-6 gap-y-1 pt-3 text-xs">
								<div>
									<span style="color: var(--oo-fg-muted);">Author:</span>
									<span style="color: var(--oo-fg-secondary);">{plugin.author}</span>
								</div>
								<div>
									<span style="color: var(--oo-fg-muted);">State:</span>
									<span style="color: {stateColor(plugin.state)};">{plugin.state}</span>
								</div>
								<div>
									<span style="color: var(--oo-fg-muted);">Installed:</span>
									<span style="color: var(--oo-fg-secondary);">{formatTimestamp(plugin.installed_at)}</span>
								</div>
								<div>
									<span style="color: var(--oo-fg-muted);">Updated:</span>
									<span style="color: var(--oo-fg-secondary);">{formatTimestamp(plugin.updated_at)}</span>
								</div>
							</div>

							<!-- Hooks -->
							{#if plugin.hooks.length > 0}
								<div>
									<span class="text-xs font-medium" style="color: var(--oo-fg-muted);">Hooks:</span>
									<div class="flex flex-wrap gap-1 mt-1">
										{#each plugin.hooks as hook}
											<span class="text-xs px-2 py-0.5 rounded-full"
												style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-secondary);
													border: 1px solid var(--oo-bd-subtle);">
												{hook}
											</span>
										{/each}
									</div>
								</div>
							{/if}

							<!-- How to use -->
							{#if usageHint(plugin)}
								{@const hint = usageHint(plugin)}
								<div class="rounded-lg px-3 py-2"
									style="background-color: var(--oo-bg-overlay); border: 1px solid var(--oo-bd-subtle);">
									<span class="text-xs font-medium" style="color: var(--oo-fg-muted);">How to use:</span>
									{#if hint?.type === 'commands'}
										<div class="flex flex-wrap gap-1.5 mt-1.5">
											{#each (hint?.detail ?? '').split('  ') as cmd}
												<code class="text-[11px] px-1.5 py-0.5 rounded font-mono"
													style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);
														border: 1px solid var(--oo-bd-subtle);">
													{cmd}
												</code>
											{/each}
										</div>
									{:else}
										<p class="text-xs mt-1" style="color: var(--oo-fg-tertiary);">
											{hint?.detail}
										</p>
									{/if}
								</div>
							{/if}

							<!-- Permissions -->
							{#if plugin.permissions.length > 0}
								<div>
									<span class="text-xs font-medium" style="color: var(--oo-fg-muted);">Permissions:</span>
									<div class="flex flex-wrap gap-1 mt-1">
										{#each plugin.permissions as perm}
											<span class="text-xs px-2 py-0.5 rounded-full"
												style="background-color: var(--oo-warning-bg); color: var(--oo-warning);
													border: 1px solid var(--oo-warning-bd);">
												{perm}
											</span>
										{/each}
									</div>
								</div>
							{/if}

							<!-- Dependencies -->
							{#if plugin.dependencies.length > 0}
								<div>
									<span class="text-xs font-medium" style="color: var(--oo-fg-muted);">Dependencies:</span>
									<span class="text-xs ml-1" style="color: var(--oo-fg-secondary);">
										{plugin.dependencies.join(', ')}
									</span>
								</div>
							{/if}

							<!-- Config editor -->
							{#if configLoading}
								<p class="text-xs" style="color: var(--oo-fg-muted);">Loading config...</p>
							{:else if pluginConfig}
								<div>
									<span class="text-xs font-medium" style="color: var(--oo-fg-muted);">Configuration:</span>
									<textarea
										bind:value={configEditorText}
										rows="5"
										class="w-full mt-1 px-3 py-2 rounded-lg text-xs font-mono"
										style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd);
											color: var(--oo-fg-primary); resize: vertical;"
										spellcheck="false"
									></textarea>
									<div class="flex gap-2 mt-2">
										<button
											on:click={handleSaveConfig}
											disabled={savingConfig}
											class="px-3 py-1.5 rounded-lg text-xs font-medium disabled:opacity-50"
											style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
										>
											{savingConfig ? 'Saving...' : 'Save Config'}
										</button>
										<button
											on:click={closeDetails}
											class="px-3 py-1.5 rounded-lg text-xs"
											style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
										>
											Close
										</button>
									</div>
								</div>

								<!-- Config schema hint -->
								{#if pluginConfig.config_schema && Object.keys(pluginConfig.config_schema).length > 0}
									<div class="text-xs" style="color: var(--oo-fg-muted);">
										<span class="font-medium">Schema:</span>
										{#each Object.entries(pluginConfig.config_schema) as [key, schema]}
											<div class="ml-2 mt-0.5">
												<span class="font-mono" style="color: var(--oo-fg-secondary);">{key}</span>
												{#if typeof schema === 'object' && schema !== null}
													<span> ({schema.type || 'any'}{schema.default !== undefined ? `, default: ${schema.default}` : ''})</span>
													{#if schema.description}
														<span> — {schema.description}</span>
													{/if}
												{/if}
											</div>
										{/each}
									</div>
								{/if}
							{/if}

							<!-- Uninstall -->
							<div class="pt-2" style="border-top: 1px solid var(--oo-bd-subtle);">
								{#if confirmUninstall === plugin.name}
									<div class="flex items-center gap-2">
										<span class="text-xs" style="color: var(--oo-error);">
											Permanently remove this plugin?
										</span>
										<button
											on:click={() => handleUninstall(plugin.name)}
											class="px-3 py-1 rounded-lg text-xs font-medium"
											style="background-color: var(--oo-error-bg); color: var(--oo-error);
												border: 1px solid var(--oo-error-bd);"
										>
											Confirm
										</button>
										<button
											on:click={() => { confirmUninstall = null; }}
											class="px-3 py-1 rounded-lg text-xs"
											style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
										>
											Cancel
										</button>
									</div>
								{:else}
									<button
										on:click={() => { confirmUninstall = plugin.name; }}
										class="text-xs px-3 py-1 rounded-lg"
										style="color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
									>
										Uninstall
									</button>
								{/if}
							</div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
	{/if}
</div>
