/**
 * API client for the SKILL.md registry (Theme 3 / Odysseus Core).
 *
 * Defines the contract the skills-manager panel (SkillsPanel.svelte) consumes to
 * browse and manage the evolving-skills registry: list published skills and the
 * drafts left from before proposals, view one exactly by its status, with its
 * text whole as shown and its digest, then publish a draft, delete a draft or a
 * published skill, or adopt a published skill's bytes -- each by the digest of
 * what the person was shown, so a text changed since is refused (409), never
 * acted on. It mirrors the backend SkillRegistry surface
 * (opti_oignon/agent/skills.py). The agent's and its teacher's own skill writes
 * are proposals in the review queue (lib/api/pendingWrites.ts), which the panel
 * reviews beside the registry.
 */

import { apiGet, apiPost, apiDelete } from './client';

/** A skill is either a draft left from before proposals or a published procedure. */
export type SkillStatus = 'draft' | 'published';

/** What a published skill's bytes are to a prompt here. */
export type PromptState = 'local' | 'adopted' | 'unadopted';

/** One skill, mirroring the agent route's skill payload. */
export interface Skill {
	name: string;
	category: string;
	status: SkillStatus;
	version: number;
	source: string;
	created_at: string;
	updated_at: string;
	/** status:category/name: a draft and the published skill of one name never share it. */
	key: string;
	/** The SHA-256 of the canonical body: the digest that publishes or deletes this text. */
	sha256: string;
	/** The canonical body; present when a single skill is fetched. */
	body?: string;
	/** The body as the approval drawer shows it, whole; present when a single skill is fetched. */
	shown?: string;
	/** A published skill's file bytes as shown, and their digest: what an adoption names. */
	raw_shown?: string;
	file_sha256?: string;
	/**
	 * For a published skill, what its bytes are to this device: written here
	 * (`local`), received from a paired device and adopted here (`adopted`),
	 * or received and never adopted (`unadopted`).
	 */
	sync_state?: PromptState;
	/**
	 * Whether a published skill's bytes may enter a prompt here: written by
	 * hand (`local`), named by their digest on this device (`adopted`), or
	 * not (`unadopted`) -- `/skill` and the agent's consultation refuse those.
	 */
	prompt_state?: PromptState;
}

/** The registry index payload. */
export interface SkillList {
	skills: Skill[];
}

/** The skills registry API surface, mounted under the agent route. */
const BASE = '/api/agent/skills';

function ref(category: string, name: string): string {
	return `${BASE}/${encodeURIComponent(category)}/${encodeURIComponent(name)}`;
}

/** List skills; includes the drafts unless told otherwise. */
export async function listSkills(includeDrafts = true): Promise<Skill[]> {
	const res = await apiGet<SkillList>(BASE, { include_drafts: String(includeDrafts) });
	return res?.skills ?? [];
}

/** Fetch exactly the draft or the published skill named, its text whole as shown, with its digest. */
export async function getSkill(category: string, name: string, status: SkillStatus): Promise<Skill> {
	return apiGet<Skill>(ref(category, name), { status });
}

/** Publish a draft as the text shown, named by its digest; a draft changed since is refused. */
export async function publishSkill(category: string, name: string, sha256: string): Promise<Skill> {
	return apiPost<Skill>(`${ref(category, name)}/publish`, { sha256 });
}

/** Delete exactly the draft or the published skill named, by the digest of its text shown. */
export async function deleteSkill(
	category: string,
	name: string,
	status: SkillStatus,
	sha256: string
): Promise<{ deleted: boolean }> {
	const query = `status=${encodeURIComponent(status)}&sha256=${encodeURIComponent(sha256)}`;
	return apiDelete<{ deleted: boolean }>(`${ref(category, name)}?${query}`);
}

/** Adopt a published skill's bytes not admitted to a prompt here, named by the digest of the bytes shown. */
export async function adoptSkill(category: string, name: string, sha256: string): Promise<Skill> {
	return apiPost<Skill>(`${ref(category, name)}/adopt`, { sha256 });
}

/** True for a draft. */
export function isDraft(skill: Skill): boolean {
	return skill.status === 'draft';
}

/** The words a delete button says: the target it deletes, never another of the same name. */
export function deleteLabel(skill: Skill): string {
	return skill.status === 'draft' ? 'Delete this draft' : `Delete published v${skill.version}`;
}
