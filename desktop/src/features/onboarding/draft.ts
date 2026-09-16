import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { detectTimezone } from '@/lib/utils'

export interface OnboardingDraft {
  user_name: string
  assistant_name: string
  timezone: string
  location: string
  professional_context: string
  personal_context: string
  persona: string
  brainMode: 'local' | 'cloud'
  primary: string
  fast: string
  embedding: string
  cloudProvider: string
}

interface DraftState extends OnboardingDraft {
  set: (patch: Partial<OnboardingDraft>) => void
  reset: () => void
}

const initial = (): OnboardingDraft => ({
  user_name: '',
  assistant_name: 'Sentient',
  timezone: detectTimezone(),
  location: '',
  professional_context: '',
  personal_context: '',
  persona: 'friendly',
  brainMode: 'local',
  primary: '',
  fast: '',
  embedding: '',
  cloudProvider: ''
})

/** Onboarding answers survive a reload (sessionStorage) until setup finishes. */
export const useOnboardingDraft = create<DraftState>()(
  persist(
    (set) => ({
      ...initial(),
      set: (patch) => set(patch),
      reset: () => set(initial())
    }),
    { name: 'sentient.onboarding', storage: createJSONStorage(() => sessionStorage) }
  )
)

export const STEPS = [
  { id: 'welcome', label: 'Welcome' },
  { id: 'about', label: 'About you' },
  { id: 'context', label: 'Your context' },
  { id: 'personality', label: 'Personality' },
  { id: 'brain', label: 'Brain' },
  { id: 'apps', label: 'Apps' },
  { id: 'done', label: 'Done' }
] as const

export type StepId = (typeof STEPS)[number]['id']
