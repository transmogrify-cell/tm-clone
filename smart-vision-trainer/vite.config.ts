import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

const stub = fileURLToPath(new URL('./src/ml/mediapipeStub.ts', import.meta.url));

// Pure static site, fully offline: every model is served from public/models.
export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: {
    alias: {
      // Only the tfjs runtimes of the face and hand detectors are used (see src/ml/mediapipeStub.ts).
      '@mediapipe/face_detection': stub,
      '@mediapipe/hands': stub,
    },
  },
  server: {
    port: 5175,
    host: 'localhost', // getUserMedia needs a secure context; localhost is one
  },
  build: {
    target: 'es2020',
    // tfjs + four model packages is ~2.5 MB minified; expected for an in-browser ML app.
    chunkSizeWarningLimit: 3500,
  },
});
