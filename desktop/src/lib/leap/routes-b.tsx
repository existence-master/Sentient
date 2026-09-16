/**
 * Screens owned by desktop agent B (knowing you and automation), registered without touching shared files.
 * App.tsx mounts LEAP_B_ROUTES inside the app shell; the sidebar appends LEAP_B_NAV.
 */
import { IconUserHeart, type Icon } from '@tabler/icons-react'
import type { ReactElement } from 'react'
import { AboutPage } from '@/pages/about/AboutPage'

export const LEAP_B_ROUTES: Array<{ path: string; element: ReactElement }> = [
  { path: '/about', element: <AboutPage /> },
  { path: '/about/:tab', element: <AboutPage /> }
]

export const LEAP_B_NAV: Array<{ to: string; label: string; icon: Icon; match: string }> = [
  { to: '/about', label: 'About you', icon: IconUserHeart, match: '/about' }
]
