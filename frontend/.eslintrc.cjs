/*
 * ESLint configuration (-- frontend lint in CI).
 *
 * Targets SvelteKit 2 + Svelte 4 + TypeScript. Uses the eslintrc format for
 * compatibility with the pinned ESLint 8 toolchain. Linting is syntactic only
 * (no type-aware rules) so it runs without a generated .svelte-kit/tsconfig;
 * type checking is handled separately by `npm run check` (svelte-check).
 */
module.exports = {
	root: true,
	env: {
		browser: true,
		es2021: true,
		node: true,
	},
	parser: '@typescript-eslint/parser',
	parserOptions: {
		ecmaVersion: 2021,
		sourceType: 'module',
		extraFileExtensions: ['.svelte'],
	},
	plugins: ['@typescript-eslint'],
	extends: [
		'eslint:recommended',
		'plugin:@typescript-eslint/recommended',
		'plugin:svelte/recommended',
	],
	overrides: [
		{
			files: ['*.svelte'],
			parser: 'svelte-eslint-parser',
			parserOptions: {
				parser: '@typescript-eslint/parser',
			},
		},
	],
	ignorePatterns: [
		'node_modules/',
		'build/',
		'.svelte-kit/',
		'dist/',
		'static/',
		'playwright-report/',
		'*.config.js',
		'*.config.ts',
	],
	rules: {
		// TypeScript handles undefined-symbol resolution; the core rule would
		// false-positive on Svelte/TS globals.
		'no-undef': 'off',
		// Allow intentional escape hatches; the codebase uses `any` sparingly.
		'@typescript-eslint/no-explicit-any': 'off',
		// Underscore-prefixed args/vars are intentional placeholders.
		'@typescript-eslint/no-unused-vars': [
			'warn',
			{ argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
		],
		// Svelte compiler *warnings* (a11y hints, unused-export-let) are
		// already surfaced by svelte-check (`npm run check`) and the build.
		// ESLint should flag only genuine Svelte compile *errors*, not
		// duplicate the compiler's stylistic warnings.
		'svelte/valid-compile': ['error', { ignoreWarnings: true }],
		// Intentional infinite loops (event pumps, stream readers) use
		// `while (true)`; do not flag the constant loop condition.
		'no-constant-condition': ['error', { checkLoops: false }],
		// Code is tab-indented; continued lines (multi-line attribute values,
		// JSDoc asterisks) use spaces for alignment. Smart-tabs permits
		// spaces *after* tabs and still flags spaces *before* tabs.
		'no-mixed-spaces-and-tabs': ['error', 'smart-tabs'],
	},
};
