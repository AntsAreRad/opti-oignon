# Interpreting Red Team Reports

## Report structure

After a red team audit completes, the report contains:

- **Overall resistance score** -- percentage of attacks successfully
  blocked (higher is better)
- **Per-target scores** -- breakdown by defense component
- **Per-category scores** -- breakdown by attack type
- **Individual attack results** -- each attack with its payload,
  strategy, target, and pass/fail status
- **Suggestions** -- recommended improvements based on detected
  weaknesses


## Reading the scores

| Score range | Interpretation |
|-------------|---------------|
| 90-100% | Strong defense, most attacks blocked |
| 70-89% | Good baseline, some gaps to address |
| 50-69% | Moderate risk, targeted improvements needed |
| Below 50% | Significant weaknesses, prioritize remediation |

The resistance score feeds into the startup security checklist (check
9: `redteam_resistance`). A score below the configured threshold
triggers a warning on every startup.


## Viewing reports

### From the CLI

```bash
# Latest report summary
oo redteam report

# Detailed JSON output
oo redteam report --format json

# Specific report by ID
oo redteam report --id <report-id>
```

### From the API

```
GET /api/security/redteam/results       # Latest results
GET /api/security/redteam/reports       # All reports
GET /api/security/redteam/reports/{id}  # Specific report
```


## Comparing reports

Compare two reports to track improvement over time:

```bash
oo redteam compare <report-id-1> <report-id-2>
```

Or via the API:

```
GET /api/security/redteam/compare?id1=<id1>&id2=<id2>
```

The comparison shows:

- Score delta per target and category
- New passes and new failures
- Regression detection (previously-blocked attacks that now succeed)


## Acting on results

### Reviewing suggestions

Reports include actionable suggestions for each failed test. You can
accept or reject suggestions via the API:

```
POST /api/security/redteam/suggestions/{id}/accept
POST /api/security/redteam/suggestions/{id}/reject
```

Accepted suggestions are recorded and can feed back into the RAG
sanitizer configuration.

### Feedback loop

The red team feedback loop analyzes failed attacks and can
automatically suggest new injection markers for `config/rag.yaml`.
This creates a continuous improvement cycle:

1. Red team discovers a bypass
2. Feedback loop suggests a new detection pattern
3. Pattern is added to the sanitizer configuration
4. Next audit verifies the fix

### Prioritizing fixes

Focus on:

1. **Targets with lowest scores** -- a target scoring 40% is more
   urgent than one at 85%
2. **Categories with repeated failures** -- patterns indicate
   systematic gaps rather than edge cases
3. **Regressions** -- attacks that previously were blocked but now
   succeed (use the compare feature to detect these)
