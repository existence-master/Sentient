/** The ask window's entry (#173): same theme and design system as the main window. */
import '@fontsource-variable/inter'
import '@fontsource/jetbrains-mono/400.css'
import '../styles/globals.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { applyTheme } from '@/lib/theme'
import { useUI } from '@/stores/ui'
import { AskApp } from './AskApp'

const { theme, accent } = useUI.getState()
applyTheme(theme, accent)

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <AskApp />
  </StrictMode>
)
