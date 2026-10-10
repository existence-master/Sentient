import { resolve } from 'node:path'
import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { Plugin } from 'vite'

/**
 * Production builds get a strict Content-Security-Policy meta tag. The backend port is
 * chosen at launch, so the meta tag allows loopback on any port; the main process adds
 * an exact-origin CSP header for every response it can intercept.
 */
function cspPlugin(): Plugin {
  const csp = [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob: http://127.0.0.1:*",
    "media-src 'self' data: blob: http://127.0.0.1:*",
    "font-src 'self' data:",
    "connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'"
    // frame-ancestors is ignored in <meta>; the window never loads in a frame anyway.
  ].join('; ')
  return {
    name: 'sentient-csp',
    apply: 'build',
    transformIndexHtml(html) {
      return html.replace(
        '<!-- CSP -->',
        `<meta http-equiv="Content-Security-Policy" content="${csp}" />`
      )
    }
  }
}

export default defineConfig({
  main: {
    build: {
      outDir: 'out/main',
      lib: { entry: resolve(__dirname, 'electron/main/index.ts') }
    }
  },
  preload: {
    build: {
      outDir: 'out/preload',
      lib: { entry: resolve(__dirname, 'electron/preload/index.ts') }
    }
  },
  renderer: {
    root: '.',
    base: './',
    publicDir: resolve(__dirname, 'public'),
    resolve: { alias: { '@': resolve(__dirname, 'src') } },
    plugins: [react(), tailwindcss(), cspPlugin()],
    server: { port: 5199, strictPort: false },
    build: {
      outDir: 'out/renderer',
      emptyOutDir: true,
      chunkSizeWarningLimit: 2000,
      // pill.html: the small listening window for push to talk and dictation (#169)
      rollupOptions: { input: { index: resolve(__dirname, 'index.html'), pill: resolve(__dirname, 'pill.html') } }
    }
  }
})
