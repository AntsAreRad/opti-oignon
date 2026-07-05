/**
 * Conversation-scoped workspace binding store.
 *
 * Holds which workspace the ACTIVE conversation is bound to, for the
 * chat surface (binding badge) and the workspace panel to share. The
 * scope is the whole point: switching conversations clears the value
 * immediately, a late answer from a conversation the user already left
 * never paints the new one (epoch guard), and bind/unbind actions
 * performed in the panel update the store only when they target the
 * conversation on screen -- an explicit action always wins over a stale
 * refresh still in flight.
 *
 * Implements the Svelte store contract (subscribe) by hand, with no
 * dependency, so the logic is exercised directly under Node without a
 * bundler and the components consume it like any other store.
 */

export interface WorkspaceBindingState {
	/** The conversation this state belongs to (null: none active). */
	conversationId: string | null;
	/** The bound workspace id, or null when unbound or unknown yet. */
	sessionId: string | null;
	/** True while a refresh for conversationId is in flight. */
	loading: boolean;
}

type Subscriber = (value: WorkspaceBindingState) => void;

/** Fetches the binding of one conversation (the API client, injected). */
export type BindingFetcher = (
	conversationId: string
) => Promise<{ session_id: string | null }>;

export class WorkspaceBindingStore {
	private value: WorkspaceBindingState = {
		conversationId: null,
		sessionId: null,
		loading: false
	};
	private readonly subscribers = new Set<Subscriber>();
	private epoch = 0;

	/** Svelte store contract: immediate emission, returns unsubscribe. */
	subscribe(run: Subscriber): () => void {
		this.subscribers.add(run);
		run(this.value);
		return () => {
			this.subscribers.delete(run);
		};
	}

	/** The current value (tests and imperative call sites). */
	get snapshot(): WorkspaceBindingState {
		return this.value;
	}

	private emit(next: WorkspaceBindingState): void {
		this.value = next;
		for (const run of this.subscribers) {
			run(next);
		}
	}

	/**
	 * Target a conversation and refresh its binding. The previous value
	 * is cleared immediately (no bleed while the fetch is in flight),
	 * and a response arriving after a later refreshFor or after an
	 * explicit apply is dropped.
	 */
	async refreshFor(
		conversationId: string | null,
		fetcher: BindingFetcher
	): Promise<void> {
		const myEpoch = ++this.epoch;
		if (!conversationId) {
			this.emit({ conversationId: null, sessionId: null, loading: false });
			return;
		}
		this.emit({ conversationId, sessionId: null, loading: true });
		let sessionId: string | null = null;
		try {
			const result = await fetcher(conversationId);
			sessionId = result?.session_id ?? null;
		} catch {
			sessionId = null;
		}
		if (myEpoch !== this.epoch) {
			return; // stale response from a conversation the user left
		}
		this.emit({ conversationId, sessionId, loading: false });
	}

	/** A bind succeeded in the panel; only the active conversation paints. */
	applyBound(conversationId: string, sessionId: string): void {
		if (this.value.conversationId !== conversationId) {
			return;
		}
		this.epoch += 1; // the explicit action wins over any in-flight refresh
		this.emit({ conversationId, sessionId, loading: false });
	}

	/** An unbind succeeded in the panel; only the active conversation paints. */
	applyUnbound(conversationId: string): void {
		if (this.value.conversationId !== conversationId) {
			return;
		}
		this.epoch += 1;
		this.emit({ conversationId, sessionId: null, loading: false });
	}
}

/** The app-wide instance the chat surface and the panel share. */
export const workspaceBinding = new WorkspaceBindingStore();
