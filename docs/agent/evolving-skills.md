# Evolving Skills

A skill is a reusable procedure -- a SKILL.md document with a When to Use section,
a Procedure, Pitfalls, and a Verification step. The agent consults skills before
domain work and proposes new or improved ones as it learns, but it can never
publish a skill on its own: every write is a draft that waits for human approval.

## The registry

Skills live in a registry on disk under `data/skills/`. Each published skill is a
`<category>/<name>/SKILL.md`; drafts live under a `.drafts/` area until approved;
prior versions are retained under `.versions/`. A `_usage.json` sidecar records
how often each skill is consulted, so a consultation never rewrites the SKILL.md
itself. Path components are strictly validated, and every mutation is appended to
the hash-chain audit log.

## Consultation

Before starting domain work the agent searches the registry for a procedure it may
already have and views the most relevant one. Skill text re-entering the prompt is
wrapped as untrusted reference -- it is information to reason about, not commands to
obey, and any forged untrusted-data marker inside a skill body is neutralised.

## Proposing and approving

The agent extends the registry through the `manage_skills` tool. Reads (list,
index, view, search) are ungated. Writes (add, edit, patch, publish, delete) go
through the tool-call approval gate, fail-secure. A new skill is added as a draft,
its procedure is exercised in the sandbox where it declares verification commands,
and it stays unpublished until a human approves it. Teacher-escalation drafts
follow the same publish-after-approval path.

The skills-manager panel in the UI is the human's review-and-approve surface: it
browses published skills and the agent-proposed drafts, lets you expand a skill to
read its procedure, and surfaces the approval-gated actions -- approve-and-publish
a draft, or delete one. Drafts are clearly marked as awaiting approval.

## API surface

The panel consumes the skills surface mounted under the agent route:

- `GET /api/agent/skills` lists published skills and drafts.
- `GET /api/agent/skills/{category}/{name}` returns a skill with its body.
- `POST /api/agent/skills/{category}/{name}/publish` approves and publishes a draft.
- `DELETE /api/agent/skills/{category}/{name}` removes a skill.
