import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    {
      name: 'research-entry-only-in-research-mode',
      transformIndexHtml: {
        order: 'pre',
        handler(html) {
          return mode === 'research'
            ? html.replace('/src/entry.production.tsx', '/src/entry.research.tsx')
            : html
        },
      },
    },
  ],
}))
