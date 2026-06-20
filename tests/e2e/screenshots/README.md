# Visual Regression Baselines

This directory is reserved for Playwright visual regression reference screenshots.

## How it works

Playwright's `toHaveScreenshot()` stores baseline images in a sibling
`-snapshots/` directory next to the spec file:

```
tests/e2e/visual.spec.ts-snapshots/
├── desktop-chat-empty-chromium.png
├── desktop-chat-messages-chromium.png
├── desktop-login-chromium.png
├── desktop-settings-quick-chromium.png
├── desktop-settings-security-chromium.png
├── mobile-chat-empty-chromium.png
├── mobile-login-chromium.png
└── ...
```

## Generate baselines

```bash
cd frontend
npx playwright test visual.spec.ts --update-snapshots
```

## Compare against baselines

```bash
cd frontend
npx playwright test visual.spec.ts
```

Thresholds (in `playwright.config.ts`):
- `maxDiffPixelRatio: 0.05` — up to 5% pixel difference tolerated
- `threshold: 0.2` — per-pixel color sensitivity

Failed comparisons produce diff images in `tests/e2e/test-results/`.
