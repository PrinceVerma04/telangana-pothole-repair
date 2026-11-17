/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// VITE_API_BASE: '' (default) = same origin; the dev server proxies /v1 to API_PROXY_TARGET (default :8000).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.API_PROXY_TARGET || 'http://localhost:8000';
  return {
    plugins: [
      react(),
      VitePWA({
        strategies: 'injectManifest',
        srcDir: 'src',
        filename: 'sw.ts',
        registerType: 'prompt',
        injectRegister: false,
        devOptions: { enabled: false },
        injectManifest: {
          globPatterns: ['**/*.{js,css,html,svg,png,woff2,webmanifest}'],
          maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
        },
        includeAssets: ['favicon.svg', 'robots.txt'],
        manifest: {
          name: 'RoadFix — Pothole Detection & Maintenance',
          short_name: 'RoadFix',
          description: 'Pothole measurement, prioritisation and repair operations for Telangana',
          lang: 'te',
          dir: 'ltr',
          start_url: '/',
          scope: '/',
          display: 'standalone',
          orientation: 'any',
          background_color: '#f5f6f8',
          theme_color: '#0b3d6e',
          categories: ['government', 'utilities', 'navigation'],
          icons: [
            { src: 'pwa-192.png', sizes: '192x192', type: 'image/png' },
            { src: 'pwa-512.png', sizes: '512x512', type: 'image/png' },
            { src: 'pwa-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
          ],
          shortcuts: [
            { name: 'Field mode', short_name: 'Field', url: '/field' },
            { name: 'Report a pothole', short_name: 'Report', url: '/citizen' },
          ],
        },
      }),
    ],
    server: {
      port: 5173,
      proxy: { '/v1': { target: apiTarget, changeOrigin: true }, '/healthz': { target: apiTarget, changeOrigin: true } },
    },
    preview: { port: 4173 },
    build: {
      target: 'es2022',
      sourcemap: false,
      chunkSizeWarningLimit: 1600,
    },
    test: {
      environment: 'node',
      include: ['src/**/*.test.ts'],
      setupFiles: ['src/test/setup.ts'],
    },
  };
});
