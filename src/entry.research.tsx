import React from 'react'
import { createRoot } from 'react-dom/client'
import { ResearchApp } from './research-app'
import { SiteHeader } from './site'
import './style.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SiteHeader research />
    <ResearchApp />
  </React.StrictMode>,
)
