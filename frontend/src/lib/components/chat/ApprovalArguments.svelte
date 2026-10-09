<!--
  ApprovalArguments.svelte
  What a person reads before allowing a held tool call: every argument the
  call would run with, its value whole as the approval queue shows it (up to
  its bound, every hidden character already written as its escape by the
  backend), where it came from when the backend says, and the value's own
  length and lines, so a tail below the box's fold is never taken for the end
  of the value. Shared by every surface that offers Allow or Deny, so none
  approves on a summary alone. When the backend sends no arguments, its
  summary is shown instead. English only.
-->
<script lang="ts">
	import type { PendingApproval } from '$lib/api/toolCallApproval';

	export let request: PendingApproval;

	// What each label says of an argument, for the person deciding.
	const LABEL_MEANING: Record<string, string> = {
		typed: 'you typed it in this turn',
		default: "the tool's own default",
		unendorsed: 'not typed by you: the model chose it',
	};

	// A table's own entry only: an argument named `constructor` must never
	// read what every object inherits.
	function own<T>(table: Record<string, T> | undefined, key: string): T | undefined {
		return table && Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;
	}

	function shownValue(value: unknown): string {
		return typeof value === 'string' ? value : JSON.stringify(value);
	}

	// The value's own length, as the queue counted it; a shown value is longer
	// than the value by its escapes. An older backend sends no sizes: the
	// shown text is counted then, and says so.
	function valueSize(request: PendingApproval, name: string): string {
		const size = own(request.sizes, name);
		if (size) return `${size.chars} characters${size.lines > 1 ? `, ${size.lines} lines` : ''}`;
		const text = shownValue(own(request.arguments, name));
		const lines = text.split('\n').length;
		return `${text.length} characters shown${lines > 1 ? `, ${lines} lines` : ''}`;
	}

	$: names = request.arguments ? Object.keys(request.arguments) : [];
</script>

{#if names.length > 0}
	<ul class="text-xs mb-3 flex flex-col gap-2" aria-label="Each argument, and where it came from">
		{#each Object.keys(request.arguments) as name (name)}
			<li class="flex flex-col gap-1">
				<span class="flex gap-2">
					<span class="font-mono text-[var(--oo-fg-primary)]">{name}</span>
					{#if own(request.labels, name)}
						<span class={own(request.labels, name) === 'unendorsed' ? 'text-[var(--oo-warning)]' : 'text-[var(--oo-fg-muted)]'}>{own(LABEL_MEANING, own(request.labels, name) ?? '') ?? own(request.labels, name)}</span>
					{/if}
					<span class="ml-auto text-[var(--oo-fg-faint)]">{valueSize(request, name)}</span>
				</span>
				<pre class="p-2 rounded whitespace-pre-wrap break-all max-h-48 overflow-auto bg-[var(--oo-bg-subtle)] text-[var(--oo-fg-primary)]">{shownValue(request.arguments[name])}</pre>
			</li>
		{/each}
	</ul>
{:else if request.arguments_summary}
	<pre class="text-xs mb-3 p-2 rounded whitespace-pre-wrap break-all bg-[var(--oo-bg-subtle)] text-[var(--oo-fg-muted)]">{request.arguments_summary}</pre>
{/if}
