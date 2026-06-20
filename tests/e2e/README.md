# E2E Tests — Opti-Oignon Frontend

Playwright-based end-to-end tests for critical user flows.

## Setup

```bash
cd frontend
npm install
npx playwright install chromium --with-deps
```

## Run

```bash
# From project root
./scripts/run_e2e.sh

# Or from frontend/
npm run test:e2e
npm run test:e2e:headed   # visible browser
npm run test:e2e:ui       # interactive UI
```

## Structure

```
tests/e2e/
├── mocks/          # API route interception mocks
├── screenshots/    # Visual regression baselines
├── auth.spec.ts    # Auth flow (login/register/logout)
├── chat.spec.ts    # Chat flow (send/receive messages)
├── settings.spec.ts # Settings navigation
├── rag.spec.ts     # RAG upload + query
├── plugins.spec.ts # Plugin lifecycle
├── security.spec.ts # Security mode + kill switch
├── mobile.spec.ts  # Mobile responsive checks
└── visual.spec.ts  # Visual regression baseline capture
```
