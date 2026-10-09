# Evolving Skills

A skill is a reusable procedure -- a SKILL.md document with a When to Use section,
a Procedure, Pitfalls, and a Verification step. The agent consults skills before
domain work and proposes new or improved ones as it learns, but it can never
publish a skill on its own: every write it makes is a proposal that waits for you,
and you publish it by naming the digest of the text you read.

## The registry

Skills live in a registry on disk under `data/skills/`. Each published skill is a
`<category>/<name>/SKILL.md`; drafts left from before proposals live under a
`.drafts/` area until published; prior versions are retained under `.versions/`.
A `_usage.json` sidecar records how often each skill is consulted, so a
consultation never rewrites the SKILL.md itself. Path components are strictly
validated, and every mutation is appended to the hash-chain audit log.

## Consultation

Before starting domain work the agent searches the registry for a procedure it may
already have and views the most relevant one. Only skills whose bytes may enter a
prompt here are consulted, by the planner and by the agent's own search and view
alike: skills written on this device by hand, and skills whose bytes you named by
their digest -- accepted from a proposal, published from a draft, or adopted. A
skill is named by the folder it lies in, never by what its text claims. Skill text
re-entering the prompt is wrapped as untrusted reference -- it is information to
reason about, not commands to obey, and any forged untrusted-data marker inside a
skill body is neutralised.

## Proposing and accepting

The agent extends the registry through the `manage_skills` tool. Reads ask no one:
list and index show names, and search and view hand it only what the consultation
would, never a draft's text. Every write -- add, edit, patch, delete -- is a
proposal in the review queue, and no one is asked while the agent runs. A proposal
holds the category and name it would be written under, the text that would be
written (for a patch, the text the patch would result in) and its SHA-256, and the
digest of the published text it would replace or delete; where the text declares
verification commands, they run in the sandbox before it is proposed. The agent's
`publish` action publishes nothing: publishing is yours. A draft from the teacher
model is proposed the same way.

You review proposals in the skills panel, or with `/review` in `oo chat`. Each one
is shown whole, never cut, with every character a screen would hide -- a zero-width
space, a direction override, a Unicode tag -- written as its escape, beside the text
it replaces and its digest. Accepting names that digest: the server writes those
bytes and no others, hashes the text again as it writes it, and refuses, writing
nothing, when the published skill changed since the proposal was made.

The skills panel keys each row by its status, so a draft and the published skill of
the same name are two rows. Expanding a row shows exactly that item; publishing a
draft, deleting a draft or a published skill, and adopting a published skill's bytes
each send the digest of what was shown, and each button names its target.

A skill the agent wrote or rewrote before proposals, approved on its name alone, is
not run by `/skill` nor consulted until you adopt its bytes by their digest: a rewrite
kept the source of the skill it replaced, so any skill the registry rewrote then --
it keeps a `.versions/` history -- waits for that one adoption.

## API surface

The panel consumes the skills surface mounted under the agent route, and the review
queue's:

- `GET /api/agent/skills` lists published skills and drafts, each with its key and
  digest.
- `GET /api/agent/skills/{category}/{name}?status=draft|published` returns exactly
  that item, its text whole and as shown, with its digest.
- `POST /api/agent/skills/{category}/{name}/publish` publishes a draft named by the
  digest of its text; a draft changed since is refused.
- `DELETE /api/agent/skills/{category}/{name}?status=...&sha256=...` deletes the
  draft or the published skill named, by that digest.
- `POST /api/agent/skills/{category}/{name}/adopt` adopts a published skill's bytes
  named by their digest.
- `GET /api/pending-writes?store=skills` lists the waiting proposals, and
  `POST /api/pending-writes/accept` accepts them with the digest of each.
