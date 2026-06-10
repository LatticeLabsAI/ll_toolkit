// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

// https://astro.build/config
// Deployed as a GitHub Pages project site: https://latticelabsai.github.io/ll_toolkit/
export default defineConfig({
	site: 'https://latticelabsai.github.io',
	base: '/ll_toolkit/',
	integrations: [
		starlight({
			// Fails the build on any broken internal link (SPEC-2 FR-20).
			plugins: [starlightLinksValidator()],
			title: 'LatticeLabs Toolkit',
			description:
				'Documentation for the LatticeLabs Toolkit — a monorepo of Python packages for CAD document processing, geometric tokenization, neural CAD models, optical CAD recognition, point clouds, and generative CAD.',
			favicon: '/favicon.svg',
			// LayerDynamics brand mark. Source SVGs are hard-black, so a white
			// variant is used in dark mode to stay visible.
			logo: {
				light: './src/assets/LayerDynamicsLogo.svg',
				dark: './src/assets/LayerDynamicsLogo-dark.svg',
				alt: 'LayerDynamics — LatticeLabs Toolkit',
			},
			customCss: ['./src/styles/theme.css'],
			lastUpdated: true,
			editLink: {
				baseUrl: 'https://github.com/LatticeLabsAI/ll_toolkit/edit/main/site/',
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/LatticeLabsAI/ll_toolkit',
				},
			],
			// The sidebar is built incrementally as content lands (see docs/specs/SPEC-2).
			// Each group autogenerates from its content directory, ordered by each page's
			// `sidebar.order` frontmatter.
			sidebar: [
				{ label: 'Get Started', items: [{ autogenerate: { directory: 'get-started' } }] },
				{ label: 'Tutorials', items: [{ autogenerate: { directory: 'tutorials' } }] },
				{ label: 'How-to Guides', items: [{ autogenerate: { directory: 'guides' } }] },
				{ label: 'Concepts', items: [{ autogenerate: { directory: 'concepts' } }] },
				{
					label: 'Packages',
					items: [
						{ label: 'cadling', items: [{ autogenerate: { directory: 'cadling' } }] },
						{ label: 'll_stepnet', items: [{ autogenerate: { directory: 'll_stepnet' } }] },
						{ label: 'geotoken', items: [{ autogenerate: { directory: 'geotoken' } }] },
						{ label: 'll_ocadr', items: [{ autogenerate: { directory: 'll_ocadr' } }] },
						{ label: 'll_gen', items: [{ autogenerate: { directory: 'll_gen' } }] },
						{ label: 'll_clouds', items: [{ autogenerate: { directory: 'll_clouds' } }] },
					],
				},
				{ label: 'Roadmap', items: [{ autogenerate: { directory: 'roadmap' } }] },
				{ label: 'Contributing', items: [{ autogenerate: { directory: 'contributing' } }] },
			],

		}),
	],
});
