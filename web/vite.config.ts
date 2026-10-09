import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { target: 'es2022', chunkSizeWarningLimit: 3000 },
})
