# Evolving Memory

The agent has a two-tier memory that lets it carry durable knowledge across
conversations without re-reading everything each time.

## Two tiers

The first tier is a compact working-memory block: a short, current summary of the
facts most relevant to the conversation, injected into the prompt. The second tier
is the full archive: everything the memory has recorded, always retrievable. The
working block is intentionally lossy; when the agent needs a detail the summary
dropped, it retrieves from the archive rather than relying on the compressed form.
This dual-layer design (compressed summary in the prompt, full archive searchable)
means compression never permanently loses information.

## How facts are formed

Durable facts are distilled from conversations -- preferences, project context,
decisions -- and stored with a category and provenance. Memory that re-enters the
prompt is wrapped as untrusted data, like any other external content, so a stored
note cannot smuggle in instructions.

## Reviewing memory

The memory-manager panel in the UI lists facts grouped by category and supports
soft-delete, restore, and edit over the `/api/memories` API. Memory is the user's
to inspect and curate; nothing about a person is surfaced uninvited, and sensitive
content is only applied when it is relevant to the task at hand.

## Relation to skills

Memory holds facts; skills hold procedures. A fact is something true about the
user or the project; a skill is a reusable how-to the agent consults before doing
similar work. See [Evolving Skills](evolving-skills.md).
