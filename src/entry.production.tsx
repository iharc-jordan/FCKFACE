import React from 'react'
import { createRoot } from 'react-dom/client'
import { SiteHeader, ResearchStatus } from './site'
import './style.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SiteHeader />
    <ResearchStatus />
  </React.StrictMode>,
)
