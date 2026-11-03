import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		// No CORS of the dev server's own: its default answers a page of any loopback port, which
		// would read the API through the proxy. The app's pages are of this origin and need none.
		cors: false,
		proxy: {
			'/api': {
				target: 'http://localhost:8001',
				// Keep the browser's own Host: the garden's routes refuse a name that is not
				// this machine's, and behind a rewritten Host they could not tell; the API takes
				// a change or a socket only from a page of the Host's own origin.
				changeOrigin: false,
				ws: true
			},
			// The coding agent's live socket, served outside /api, through the same origin.
			'/ws': {
				target: 'http://localhost:8001',
				changeOrigin: false,
				ws: true
			}
		}
	},
	build: {
		// Code splitting for heavy feature modules
		rollupOptions: {
			output: {
				manualChunks(id) {
					// RAG / Knowledge base components
					if (id.includes('/rag/') || id.includes('KnowledgeBase') || id.includes('RAGDashboard')) {
						return 'chunk-rag';
					}
					// Benchmark components
					if (id.includes('Benchmark') || id.includes('benchmark')) {
						return 'chunk-benchmark';
					}
					// Coding agent components
					if (id.includes('CodingAgent') || id.includes('CodePanel') || id.includes('SandboxFile')) {
						return 'chunk-coding';
					}
					// Settings sub-panels (loaded per tab)
					if (id.includes('/settings/') && id.includes('.svelte')) {
						return 'chunk-settings';
					}
					// Telemetry / observability
					if (id.includes('Telemetry') || id.includes('Profiler') || id.includes('Observability')) {
						return 'chunk-telemetry';
					}
					// Plugin marketplace
					if (id.includes('Plugin') || id.includes('plugin')) {
						return 'chunk-plugins';
					}
					// Security panels
					if (id.includes('Security') || id.includes('Hardening') || id.includes('AuditChain')
						|| id.includes('RemoteAccess') || id.includes('KeyCeremony')) {
						return 'chunk-security';
					}
				}
			}
		},
		// Tree-shaking: ensure unused exports are eliminated
		target: 'es2020',
		minify: 'esbuild',
		chunkSizeWarningLimit: 500,
	}
});
