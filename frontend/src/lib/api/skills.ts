/**
 * API client for the SKILL.md registry (S177, Theme 3 / Odysseus Core).
 *
 * Defines the contract the skills-manager panel (SkillsPanel.svelte) consumes to
 * browse and manage the evolving-skills registry: list published skills and the
 * drafts the agent proposes, view a skill's procedure, and run the
 * approval-gated write actions -- publishing a draft (the human approval that
 * turns an agent proposal into a published skill) and deleting one. It mirrors
 * the backend SkillRegistry surface (opti_oignon/agent/skills.py). The agent's
 * own in-loop writes go through the manage_skills tool behind the tool-call
 * approval gate; this panel is the human's review-and-approve surface for the
 * drafts those proposals leave behind. The live backend route is wired during
 * the end-to-end integration; this client defines the contract the panel uses.
 */

import { apiGet, apiPost, apiDelete } from './client';

/** A skill is either an unpublished draft or a published procedure. */
export type SkillStatus = 'draft' | 'published';

/** One skill, mirroring skills.Skill.to_dict(). */
export interface Skill {
	name: string;
	category: string;
	status: SkillStatus;
	version: number;
	source: string;
	created_at: string;
	updated_at: string;
	/** The full SKILL.md body; present when a single skill is fetched. */
	body?: string;
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

/** List skills; includes agent-proposed drafts unless told otherwise. */
export async function listSkills(includeDrafts = true): Promise<Skill[]> {
	const res = await apiGet<SkillList>(BASE, { include_drafts: String(includeDrafts) });
	return res?.skills ?? [];
}

/** Fetch a single skill, including its full body. */
export async function getSkill(category: string, name: string): Promise<Skill> {
	return apiGet<Skill>(ref(category, name));
}

/**
 * Publish a draft. This is the human approval step: it promotes an
 * agent-proposed draft into a published skill the agent may then consult.
 */
export async function publishSkill(category: string, name: string): Promise<Skill> {
	return apiPost<Skill>(`${ref(category, name)}/publish`);
}

/** Delete a skill (draft or published). */
export async function deleteSkill(
	category: string,
	name: string
): Promise<{ deleted: boolean }> {
	return apiDelete<{ deleted: boolean }>(ref(category, name));
}

/** True for an unpublished, approval-pending draft. */
export function isDraft(skill: Skill): boolean {
	return skill.status === 'draft';
}
