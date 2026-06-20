/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	darkMode: 'class',
	theme: {
		extend: {
			colors: {
				surface: {
					50:  '#F5F0EB',
					100: '#EDE8E2',
					200: '#E5DFD8',
					300: '#DCD5CC',
					400: '#D4CBC2',
					500: '#ADA49B',
					600: '#8C8279',
					700: '#5C544C',
					800: '#353230',
					900: '#2A2725',
					950: '#1A1816'
				},
				accent: {
					50:  '#FDF6EE',
					100: '#F8E8D4',
					200: '#E8C9A0',
					300: '#D4AC78',
					400: '#C99A6D',
					500: '#B07D56',
					600: '#9A6B45',
					700: '#7D5636',
					800: '#614328',
					900: '#4A321E'
				}
			},
			fontFamily: {
				sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
				mono: ['JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', 'Consolas', 'monospace']
			},
			boxShadow: {
				'soft-sm': '0 1px 2px rgba(0, 0, 0, 0.08)',
				'soft-md': '0 4px 6px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.06)',
				'soft-lg': '0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.06)',
			}
		}
	},
	plugins: []
};
