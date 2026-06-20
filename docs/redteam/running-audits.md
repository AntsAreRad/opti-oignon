# Running Red Team Audits

## Overview

The red team engine is an LLM-powered security testing system that
audits Opti-Oignon's own defense layers. It generates adversarial
inputs, sends them to target components, and scores the results.

All testing runs locally using your Ollama models. No data is sent
externally.


## What gets tested

The red team engine tests these defense components:

- **RAGSanitizer** -- prompt injection filtering in ingested documents
- **RAGAugmenter** -- injection filtering in retrieved context
- **SearchResultSanitizer** -- web search result sanitization
- **PIISanitizer** -- PII detection and redaction
- **SandboxTarget** -- sandbox escape resistance
- **ChatTarget** -- direct chat injection and jailbreak resistance


## Attack categories

Attacks are organized by category:

- **injection** -- prompt injection and instruction override attempts
- **jailbreak** -- attempts to bypass safety guidelines
- **exfiltration** -- attempts to leak system prompts or internal data
- **privilege_escalation** -- attempts to gain unauthorized access


## Attack strategies

Each attack can be transformed through obfuscation strategies to test
robustness:

- **base64_encode** -- encodes payloads in Base64
- **roleplay** -- wraps attacks in fictional scenarios
- **multilingual** -- translates attacks to other languages
- **payload_splitting** -- splits the attack across multiple messages
- **character_substitution** -- uses Unicode lookalikes


## Running an audit

### From the UI

1. Go to **Settings > Advanced > Security > Red Team**
2. Select categories, strategies, and targets (or leave all selected)
3. Click **Run Audit**
4. Monitor progress in the dashboard

### From the CLI

```bash
# Full audit with default settings
oo redteam run

# Quick audit (reduced attack count)
oo redteam run --quick

# Specific categories and targets
oo redteam run --categories injection,jailbreak --targets rag_sanitizer
```

### From the API

```
POST /api/security/redteam/run
{
  "categories": ["injection", "jailbreak"],
  "strategies": ["base64_encode", "roleplay"],
  "targets": ["rag_sanitizer", "chat"],
  "attacks_per_category": 10
}
```

All fields are optional. Omitting a field uses all enabled values from
the configuration.


## Monitoring progress

```bash
oo redteam status
```

Or poll the API:

```
GET /api/security/redteam/status
```

The response includes total steps, completed steps, current category,
and current target.


## Automated scheduling

The security scheduler can run red team audits on a configurable
timer (e.g., daily, weekly). Configure quiet hours to avoid running
during active use.

The scheduler also monitors dependency vulnerabilities via pip-audit
and detects regressions when the resistance score drops below a
threshold.
