# Branch Protection Rules

Configuration guide for GitHub branch protection on the `main` branch.

## Required Status Checks

These checks must pass before a pull request can be merged into `main`:

| Check | Job name | Blocking | Notes |
|-------|----------|----------|-------|
| Lint (Python 3.11) | `Lint (Python 3.11)` | Yes | ruff check on opti_oignon/ and tests/ |
| Lint (Python 3.12) | `Lint (Python 3.12)` | Yes | Same, on Python 3.12 |
| Typecheck (Python 3.11) | `Typecheck (Python 3.11)` | Yes | mypy baseline gate |
| Typecheck (Python 3.12) | `Typecheck (Python 3.12)` | Yes | Same, on Python 3.12 |
| Test (Python 3.11) | `Test (Python 3.11)` | Yes | pytest with coverage gate |
| Test (Python 3.12) | `Test (Python 3.12)` | Yes | Same, on Python 3.12 |
| E2E Tests | `E2E Tests` | Yes | Playwright frontend tests |
| Security Scan | `Security Scan` | No | Informational only (continue-on-error) |

## Setup Instructions

In the GitHub repository settings, navigate to
**Settings > Branches > Branch protection rules > Add rule**.

### Rule configuration

- **Branch name pattern:** `main`
- **Require a pull request before merging:** enabled
  - Require approvals: 1 (adjust for team size)
  - Dismiss stale pull request approvals when new commits are pushed: enabled
- **Require status checks to pass before merging:** enabled
  - **Require branches to be up to date before merging:** enabled
  - Add the following status checks:
    - `Lint (Python 3.11)`
    - `Lint (Python 3.12)`
    - `Typecheck (Python 3.11)`
    - `Typecheck (Python 3.12)`
    - `Test (Python 3.11)`
    - `Test (Python 3.12)`
    - `E2E Tests`
- **Require conversation resolution before merging:** enabled
- **Do not allow bypassing the above settings:** recommended for
  multi-contributor projects; can be relaxed for solo development

### Security scan

The security scan job (`Security Scan`) runs with `continue-on-error: true`
in the CI workflow. This means it will not block merges even if vulnerabilities
are found. This is intentional for the initial rollout to avoid blocking
development on third-party dependency issues.

To promote security scan to a blocking check later, remove
`continue-on-error: true` from the security job in `.github/workflows/ci.yml`
and add `Security Scan` to the required status checks list above.

## Coverage Gate

The test job runs pytest with `--cov-fail-under` set via
`scripts/run_coverage.sh`. The current minimum overall threshold is 30%.
Per-module thresholds for security-critical modules are enforced separately
(see `scripts/run_coverage.sh` for the full list).

The coverage badge in the README is updated automatically on every push to
`main` by the `badge` job in the CI workflow.

## Merge Strategy

Recommended merge strategy for pull requests:

- **Squash and merge** for feature branches (clean linear history)
- **Merge commit** for release branches (preserves branch structure)
- **Rebase and merge** avoided (rewrites commit hashes, breaks signed commits)

## Workflow Dependencies

```
lint -----> test -----> badge (main only)
typecheck -/
lint -----> e2e
security (independent, non-blocking)
```

Lint and typecheck run in parallel. Test depends on both passing (fail-fast).
E2E depends on lint only. Security runs independently. Badge generation only
runs on pushes to main after tests pass.
