import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		proxy: {
			'/api': {
				target: 'http://localhost:8001',
				changeOrigin: true,
				ws: true
			}
		}
	},
	build: {
		// S134: Code splitting for heavy feature modules
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
