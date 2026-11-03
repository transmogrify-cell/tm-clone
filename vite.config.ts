import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Pure static site: no proxy, no API. `base: './'` lets the production build
// be opened from any sub-path (or a file server) without rewriting URLs.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    // getUserMedia needs a secure context; localhost counts as one.
    host: 'localhost',
  },
  build: {
    target: 'es2020',
    // tfjs is ~1.5 MB minified; that is expected for an in-browser ML app.
    chunkSizeWarningLimit: 2500,
  },
});
