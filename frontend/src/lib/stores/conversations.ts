/**
 * Svelte stores for conversation state management.
 *
 * Provides reactive stores and action functions for conversation CRUD.
 */

import { writable, derived, get } from 'svelte/store';
import type { ConversationSummary, ConversationDetail, MessageItem } from '$lib/types';
import * as api from '$lib/api/conversations';

// -- Stores --

/** Liste des conversations (sidebar). */
export const conversations = writable<ConversationSummary[]>([]);

/** Active conversation ID. */
export const activeConversationId = writable<string | null>(null);

/** Messages of the active conversation. */
export const messages = writable<MessageItem[]>([]);

/** Chargement en cours. */
export const loading = writable<boolean>(false);

/** Chargement des messages en cours. */
export const messagesLoading = writable<boolean>(false);

/** Erreur courante. */
export const error = writable<string | null>(null);

/** Conversation active derivee. */
export const activeConversation = derived(
	[conversations, activeConversationId],
	([$conversations, $activeId]) => {
		if (!$activeId) return null;
		return $conversations.find((c) => c.id === $activeId) ?? null;
	}
);

// -- Actions --

/** Charge la liste des conversations depuis l'API. */
export async function loadConversations(): Promise<void> {
	loading.set(true);
	error.set(null);
	try {
		const list = await api.listConversations({ limit: 100 });
		conversations.set(list);
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to load conversations';
		error.set(msg);
		conversations.set([]);
	} finally {
		loading.set(false);
	}
}

/** Select a conversation and load its messages. */
export async function selectConversation(id: string): Promise<void> {
	activeConversationId.set(id);
	error.set(null);
	messagesLoading.set(true);
	try {
		const msgs = await api.getMessages(id);
		messages.set(msgs);
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to load messages';
		error.set(msg);
		messages.set([]);
	} finally {
		messagesLoading.set(false);
	}
}

/** Create a new conversation and select it. Return the ID. */
export async function createNewConversation(data?: {
	title?: string;
	model?: string;
	preset?: string;
}): Promise<string> {
	error.set(null);
	try {
		const conv = await api.createConversation(data);
		conversations.update((list) => [conv, ...list]);
		activeConversationId.set(conv.id);
		messages.set([]);
		return conv.id;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to create conversation';
		error.set(msg);
		throw err;
	}
}

/** Renomme une conversation. */
export async function renameConv(id: string, title: string): Promise<void> {
	error.set(null);
	try {
		await api.renameConversation(id, title);
		conversations.update((list) =>
			list.map((c) => (c.id === id ? { ...c, title } : c))
		);
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to rename conversation';
		error.set(msg);
		throw err;
	}
}

/** Delete a conversation. */
export async function deleteConv(id: string): Promise<void> {
	error.set(null);
	try {
		await api.deleteConversation(id);
		conversations.update((list) => list.filter((c) => c.id !== id));

		// Si la conversation active est supprimee, deselectionner
		const currentId = get(activeConversationId);
		if (currentId === id) {
			activeConversationId.set(null);
			messages.set([]);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to delete conversation';
		error.set(msg);
		throw err;
	}
}
