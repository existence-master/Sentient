// Renderer-only dev server for working in a normal browser:
//   npm run dev:web  ->  http://localhost:5199/?api=http://127.0.0.1:7777&token=<token>
import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  root: '.',
  base: './',
  resolve: { alias: { '@': resolve(__dirname, 'src') } },
  plugins: [react(), tailwindcss()],
  server: { port: 5199 }
})
