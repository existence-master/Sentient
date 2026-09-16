import { IconArrowLeft, IconArrowRight, IconCheck } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { useNavigate, useParams } from 'react-router'
import { toast } from 'sonner'
import { TitleBar } from '@/components/shell/TitleBar'
import { Button } from '@/components/ui'
import { useConfig, useOnboarding } from '@/hooks/core'
import { useSetRoles } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import type { RoleName } from '@/lib/types'
import { cn } from '@/lib/utils'
import { BrainStep } from './BrainStep'
import { STEPS, useOnboardingDraft, type StepId } from './draft'
import { AboutStep, AppsStep, ContextStep, DoneStep, PersonalityStep, WelcomeStep } from './Steps'

const PROGRESS_STEPS = STEPS.filter((s) => s.id !== 'welcome' && s.id !== 'done')

export function OnboardingPage() {
  const { step: stepParam } = useParams()
  const navigate = useNavigate()
  const step: StepId = (STEPS.find((s) => s.id === stepParam)?.id ?? 'welcome') as StepId
  const index = STEPS.findIndex((s) => s.id === step)
  const draft = useOnboardingDraft()
  const onboarding = useOnboarding()
  const setRoles = useSetRoles()
  const config = useConfig()
  const [nameError, setNameError] = useState(false)
  const [direction, setDirection] = useState(1)

  const go = (id: StepId) => {
    setDirection(STEPS.findIndex((s) => s.id === id) >= index ? 1 : -1)
    navigate(`/onboarding/${id}`)
  }
  const back = () => index > 0 && go(STEPS[index - 1].id)

  const saveRoles = async () => {
    const roles = config.data?.models.roles
    const patch: Partial<Record<RoleName, string>> = {}
    for (const r of ['primary', 'fast', 'embedding'] as const) {
      const v = draft[r].trim()
      if (v && v !== roles?.[r]) patch[r] = v
    }
    if (Object.keys(patch).length) await setRoles.mutateAsync(patch)
  }

  const finish = async () => {
    if (!draft.user_name.trim()) {
      setNameError(true)
      toast.message('One more thing', { description: 'Tell Sentient what to call you.' })
      go('about')
      return
    }
    try {
      await onboarding.mutateAsync({
        user_name: draft.user_name.trim(),
        assistant_name: draft.assistant_name.trim() || 'Sentient',
        timezone: draft.timezone || 'auto',
        location: draft.location.trim(),
        professional_context: draft.professional_context.trim(),
        personal_context: draft.personal_context.trim(),
        persona: draft.persona
      })
      go('done')
    } catch (err) {
      toast.error("Couldn't finish setup", { description: errorMessage(err) })
    }
  }

  const next = async () => {
    switch (step) {
      case 'about':
        if (!draft.user_name.trim()) {
          setNameError(true)
          return
        }
        break
      case 'brain':
        try {
          await saveRoles()
        } catch (err) {
          toast.error("Couldn't save your model choice", { description: errorMessage(err) })
          return
        }
        break
      case 'apps':
        await finish()
        return
    }
    if (index < STEPS.length - 1) go(STEPS[index + 1].id)
  }

  const canSkip = step === 'context' || step === 'apps' || step === 'personality'
  const progressIndex = PROGRESS_STEPS.findIndex((s) => s.id === step)

  return (
    <div className="flex h-full flex-col bg-bg">
      <TitleBar minimal />
      <div className="relative min-h-0 flex-1 overflow-y-auto">
        <div
          aria-hidden
          className="pointer-events-none fixed left-1/2 top-[-240px] size-[760px] -translate-x-1/2 rounded-full opacity-[0.06] blur-3xl"
          style={{ background: 'radial-gradient(circle, var(--accent), transparent 65%)' }}
        />
        <div
          className="relative mx-auto flex min-h-full w-full max-w-[680px] flex-col px-8 pb-8 pt-6"
          onKeyDown={(e) => {
            const target = e.target as HTMLElement
            if (e.key === 'Enter' && !e.shiftKey && target.tagName !== 'TEXTAREA' && !target.closest('[role="listbox"],[cmdk-root],[role="dialog"]') && step !== 'welcome' && step !== 'done') {
              e.preventDefault()
              void next()
            }
          }}
        >
          {progressIndex >= 0 && (
            <div className="mb-10">
              <div className="flex gap-1.5">
                {PROGRESS_STEPS.map((s, i) => (
                  <button
                    key={s.id}
                    type="button"
                    aria-label={s.label}
                    onClick={() => i < progressIndex && go(s.id)}
                    className={cn('h-1 flex-1 rounded-full transition-colors duration-300', i <= progressIndex ? 'bg-accent' : 'bg-active', i < progressIndex && 'cursor-pointer')}
                  />
                ))}
              </div>
              <div className="mt-2.5 flex justify-between text-xs text-fg-subtle">
                <span>
                  Step {progressIndex + 1} of {PROGRESS_STEPS.length}
                </span>
                <span className="font-medium text-fg-muted">{PROGRESS_STEPS[progressIndex].label}</span>
              </div>
            </div>
          )}

          <div className="flex flex-1 flex-col">
            <AnimatePresence mode="wait" custom={direction} initial={false}>
              <motion.div
                key={step}
                custom={direction}
                initial={{ opacity: 0, x: 18 * direction }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -18 * direction }}
                transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
                className="flex flex-1 flex-col"
              >
                {step === 'welcome' && <WelcomeStep onStart={() => go('about')} />}
                {step === 'about' && <AboutStep nameError={nameError} clearNameError={() => setNameError(false)} />}
                {step === 'context' && <ContextStep />}
                {step === 'personality' && <PersonalityStep />}
                {step === 'brain' && <BrainStep />}
                {step === 'apps' && <AppsStep />}
                {step === 'done' && <DoneStep />}
              </motion.div>
            </AnimatePresence>
          </div>

          {step !== 'welcome' && step !== 'done' && (
            <div className="mt-10 flex items-center gap-2 border-t border-border pt-5">
              <Button variant="ghost" leftIcon={<IconArrowLeft size={15} />} onClick={back}>
                Back
              </Button>
              <div className="flex-1" />
              {canSkip && (
                <Button variant="ghost" onClick={() => (step === 'apps' ? void finish() : go(STEPS[index + 1].id))} disabled={onboarding.isPending}>
                  {step === 'apps' ? 'Skip for now' : 'Skip'}
                </Button>
              )}
              <Button
                variant="primary"
                size="lg"
                loading={onboarding.isPending || setRoles.isPending}
                rightIcon={step === 'apps' ? <IconCheck size={16} /> : <IconArrowRight size={16} />}
                onClick={() => void next()}
              >
                {step === 'apps' ? 'Finish setup' : 'Continue'}
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
