/**
 * The user turn as the server saves it: the words typed, then each attached
 * file under a line naming it.
 *
 * The executor (opti_oignon/executor.py, compose_user_turn) joins a turn
 * exactly so, and a contract holds the two to the same text. The chat store
 * shows the user's message this way, so it reads the same before and after a
 * reload. Pure and dependency-free: it runs under Node as it is.
 */

const HEAD = '\n\n---\nDocument provided:';

/** The text of a user turn: the words, then each document after a line naming it. */
export function userTurnText(
	message: string,
	documents: readonly { filename: string; content: string }[] = []
): string {
	let text = message;
	for (const { filename, content } of documents) {
		text += filename ? `${HEAD} ${filename}\n` : `${HEAD}\n`;
		text += content;
	}
	return text;
}
