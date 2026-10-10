import { IconAlertTriangle, IconCircleCheck, IconCloud, IconCpu, IconDeviceDesktop, IconDownload, IconExternalLink, IconEye, IconEyeOff, IconKey, IconRefresh } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, Field, IconButton, Input, ProgressBar, SegmentedControl, Skeleton } from '@/components/ui'
import { InstructionsGuide } from '@/features/integrations/InstructionsGuide'
import { CLAUDE_PLAN_STEPS, NOUS_STEPS, OpenRouterConnect } from '@/features/models/ConnectPlans'
import { ModelCheckup } from '@/features/models/ModelCheckup'
import { ModelPicker } from '@/features/models/ModelPicker'
import { ModelTest } from '@/features/models/ModelTest'
import { OllamaPull } from '@/features/models/OllamaPull'
import { useConfig } from '@/hooks/core'
import { useHardware, useLocalModels, useOllamaPull, useProviders, useSetSecret } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import { looksLikeEmbedding, recommendFastLocal, recommendLocal } from '@/lib/models'
import type { RoleName } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useOnboardingDraft } from './draft'
import { StepHeader } from './Steps'

export function BrainStep() {
  const d = useOnboardingDraft()
  const local = useLocalModels()
  const config = useConfig()
  const hardware = useHardware()

  // Pre-fill sensible defaults once we know what's installed: the model sized for this computer when it's there.
  useEffect(() => {
    if (!local.data || d.primary || hardware.isLoading) return
    const roles = config.data?.models.roles
    const rec = hardware.data?.recommendation
    // a chat-only fallback (cloud_first) is never picked for the user
    const fits = rec && !rec.cloud_first && isInstalled(rec.name, local.data.ollama.models.map((m) => m.name)) ? rec : null
    const primary = fits?.model ?? recommendLocal(local.data) ?? roles?.primary ?? ''
    d.set({
      primary,
      fast: fits?.model ?? recommendFastLocal(local.data) ?? primary,
      embedding: recommendLocal(local.data, true) ?? roles?.embedding ?? '',
      context_length: rec && rec.tier !== 'unknown' && !rec.cloud_first ? rec.context_length : null
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [local.data, config.data, hardware.isLoading])

  return (
    <div>
      <StepHeader
        title="Choose my brain"
        subtitle="Run a model privately on this computer, or use a cloud provider with your own key. You can change this any time."
      />
      <SegmentedControl
        fullWidth
        value={d.brainMode}
        onChange={(brainMode) => d.set({ brainMode })}
        options={[
          { value: 'local', label: 'On this computer', icon: <IconDeviceDesktop size={15} /> },
          { value: 'cloud', label: 'Cloud provider', icon: <IconCloud size={15} /> }
        ]}
        className="mb-5"
      />
      {d.brainMode === 'local' ? <LocalBrain /> : <CloudBrain />}
    </div>
  )
}

function LocalBrain() {
  const d = useOnboardingDraft()
  const local = useLocalModels()
  const hardware = useHardware()
  const ollama = local.data?.ollama
  const chatModels = (ollama?.models ?? []).filter((m) => !(m.is_embedding ?? looksLikeEmbedding(m.name)))
  const embedModels = (ollama?.models ?? []).filter((m) => m.is_embedding ?? looksLikeEmbedding(m.name))
  const [showPull, setShowPull] = useState(false)

  if (local.isLoading) {
    return (
      <Card className="space-y-4 p-5">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
      </Card>
    )
  }

  if (!ollama?.reachable) {
    return (
      <div className="space-y-4">
        <RecommendedForThisComputer installed={null} />
        <Alert tone="warning" icon={<IconAlertTriangle />} title="Ollama isn't running">
          Sentient uses Ollama to run models privately on your computer. Install it, open it once, then check again.
        </Alert>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" leftIcon={<IconDownload size={15} />} onClick={() => void getBridge().openExternal('https://ollama.com/download')}>
            Install Ollama
          </Button>
          <Button leftIcon={<IconRefresh size={15} className={cn(local.isFetching && 'animate-spin')} />} onClick={() => void local.refetch()}>
            Check again
          </Button>
          <Button variant="ghost" onClick={() => d.set({ brainMode: 'cloud' })}>
            Use a cloud provider instead
          </Button>
        </div>
      </div>
    )
  }

  if (!chatModels.length) {
    return (
      <div className="space-y-4">
        <RecommendedForThisComputer installed={(ollama.models ?? []).map((m) => m.name)} />
        <Card className="p-5">
          <div className="flex items-center gap-2 text-sm font-medium text-fg">
            <IconCircleCheck size={17} className="text-success" /> Ollama is running
          </div>
          <p className="mt-1 text-sm text-fg-muted">
            {hardware.data?.recommendation.cloud_first ? 'Now download a model, or use a cloud provider as recommended above.' : 'Now download a model. The one recommended above fits this computer.'}
          </p>
          <OllamaPull className="mt-4" installed={(ollama.models ?? []).map((m) => m.name)} />
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <RecommendedForThisComputer installed={(ollama.models ?? []).map((m) => m.name)} />
      <Card className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <IconCircleCheck size={17} className="text-success" />
          <span className="text-sm font-medium text-fg">Ollama detected</span>
          <Badge size="xs" tone="success">
            {chatModels.length} chat model{chatModels.length === 1 ? '' : 's'}
          </Badge>
          <div className="flex-1" />
          <Button size="xs" variant="ghost" onClick={() => setShowPull((s) => !s)}>
            {showPull ? 'Hide' : 'Download another model'}
          </Button>
        </div>
        <div className="space-y-4">
          <Field label="Main model" description="Talks with you and uses tools. We picked the best one you have installed.">
            <div className="flex items-center gap-2">
              <ModelPicker value={d.primary} onChange={(primary) => d.set({ primary })} className="flex-1" />
              <ModelTest model={d.primary} role="primary" className="shrink-0" />
            </div>
          </Field>
          <Field label="Background model" description="Handles memory, summaries and triage quietly. A smaller model keeps things snappy.">
            <ModelPicker value={d.fast} onChange={(fast) => d.set({ fast })} />
          </Field>
          <Field
            label="Memory search"
            description={embedModels.length ? 'An embedding model lets me find relevant memories.' : 'No embedding model installed yet. nomic-embed-text is small and works well.'}
          >
            <div className="flex items-center gap-2">
              <ModelPicker embedding value={d.embedding} onChange={(embedding) => d.set({ embedding })} className="flex-1" />
              <ModelTest embedding model={d.embedding} className="shrink-0" />
            </div>
          </Field>
        </div>
        {(showPull || !embedModels.length) && (
          <div className="mt-5 border-t border-border pt-4">
            <OllamaPull installed={(ollama.models ?? []).map((m) => m.name)} />
          </div>
        )}
      </Card>
      <DraftCheckup />
    </div>
  )
}

function isInstalled(name: string, installed: string[]): boolean {
  return installed.some((n) => n === name || n === `${name}:latest`)
}

/**
 * "Recommended for this computer" (#131): the local model and context length that fit its memory and graphics card,
 * detected by the engine before anything is installed. `installed` is null while Ollama isn't running.
 */
function RecommendedForThisComputer({ installed }: { installed: string[] | null }) {
  const d = useOnboardingDraft()
  const hardware = useHardware()
  const pull = useOllamaPull()
  const rec = hardware.data?.recommendation
  const use = () => rec && d.set({ primary: rec.model, fast: rec.model, context_length: rec.tier === 'unknown' ? null : rec.context_length })

  // A finished download is the model to use.
  useEffect(() => {
    if (pull.done) use()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pull.done])

  if (hardware.isLoading) return <Skeleton className="h-20 w-full rounded-xl" />
  if (!hardware.data || !rec) return null
  const summary = hardware.data.summary
  const have = installed !== null && isInstalled(rec.name, installed)
  const chosen = d.primary === rec.model && (rec.tier === 'unknown' || d.context_length === rec.context_length)
  const local =
    installed !== null &&
    (chosen ? (
      <Badge size="xs" tone="success">
        In use
      </Badge>
    ) : have ? (
      <Button size="sm" variant={rec.cloud_first ? 'ghost' : 'secondary'} onClick={use}>
        {rec.cloud_first ? 'Use for chat only' : 'Use this'}
      </Button>
    ) : (
      <Button size="sm" variant={rec.cloud_first ? 'ghost' : 'primary'} leftIcon={<IconDownload size={14} />} loading={pull.running} onClick={() => void pull.pull(rec.name)}>
        {rec.cloud_first ? 'Download for chat only' : 'Download'}
      </Button>
    ))

  if (rec.cloud_first) {
    // too little memory for a local model that can do tasks: a cloud model first, the small one only as a labelled fallback
    return (
      <Card className="p-4">
        <div className="flex items-start gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-accent-text">
            <IconCloud size={18} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-fg">Recommended for this computer: a cloud model</div>
            <p className="mt-0.5 text-xs text-fg-subtle">This computer: {summary}.</p>
            <p className="mt-1.5 text-xs leading-relaxed text-fg-muted">{rec.note}</p>
            <Button size="sm" variant="primary" className="mt-3" leftIcon={<IconCloud size={14} />} onClick={() => d.set({ brainMode: 'cloud' })}>
              Use a cloud provider
            </Button>
            <div className="mt-4 flex items-center gap-3 border-t border-border pt-3">
              <div className="min-w-0 flex-1 text-xs text-fg-subtle">
                <span className="font-medium text-fg-muted">Chat only:</span> <span className="font-mono">{rec.name}</span> runs here but can&apos;t do tasks
                reliably.
              </div>
              {local}
            </div>
            {(pull.running || pull.error) && (
              <div className="mt-3 space-y-1.5">
                <div className={cn('text-xs', pull.error ? 'text-danger' : 'text-fg-subtle')}>{pull.error ?? pull.status}</div>
                {!pull.error && <ProgressBar value={pull.progress} />}
              </div>
            )}
          </div>
        </div>
      </Card>
    )
  }

  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-accent-text">
          <IconCpu size={18} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-fg">Recommended for this computer</div>
          <p className="mt-0.5 text-xs text-fg-subtle">
            {summary === 'unknown' ? "Sentient couldn't check this computer's memory." : `This computer: ${summary}.`}
          </p>
          <p className="mt-2 text-sm text-fg">
            <span className="font-mono">{rec.name}</span>, reading {rec.context_length.toLocaleString()} tokens at a time
          </p>
          <p className="mt-0.5 text-xs leading-relaxed text-fg-muted">{rec.note}</p>
          {installed === null && <p className="mt-1 text-xs text-fg-subtle">Install Ollama first, then download it here.</p>}
          {(pull.running || pull.error) && (
            <div className="mt-3 space-y-1.5">
              <div className={cn('text-xs', pull.error ? 'text-danger' : 'text-fg-subtle')}>{pull.error ?? pull.status}</div>
              {!pull.error && <ProgressBar value={pull.progress} />}
            </div>
          )}
        </div>
        {local}
      </div>
    </Card>
  )
}

/** The model check-up for the models picked so far (they are saved when onboarding finishes). */
function DraftCheckup() {
  const d = useOnboardingDraft()
  if (!d.primary) return null
  const roles: Partial<Record<RoleName, string>> = { primary: d.primary, fast: d.fast || d.primary }
  if (d.embedding) roles.embedding = d.embedding
  return <ModelCheckup roles={roles} onUseModel={(role, model) => (role === 'primary' || role === 'fast') && d.set({ [role]: model })} />
}

function CloudBrain() {
  const d = useOnboardingDraft()
  const providers = useProviders()
  const setSecret = useSetSecret()
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  // a ChatGPT plan is signed in from Settings > Models once setup is done; its models come from the plan's own list
  const cloud = (providers.data ?? []).filter((p) => p.kind === 'cloud' && !p.sign_in)
  const selected = cloud.find((p) => p.id === d.cloudProvider)

  useEffect(() => {
    if (!d.cloudProvider && cloud.length) d.set({ cloudProvider: cloud[0].id })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloud.length])

  const choose = (id: string) => {
    const p = cloud.find((x) => x.id === id)
    const firstChat = p?.suggested.find((s) => !looksLikeEmbedding(s)) ?? ''
    d.set({ cloudProvider: id, primary: firstChat, fast: p?.suggested.filter((s) => !looksLikeEmbedding(s))[1] ?? firstChat })
    setKey('')
  }

  if (providers.isLoading) {
    return (
      <div className="grid grid-cols-4 gap-2">
        {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
          <Skeleton key={i} className="h-16 rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-2">
        {cloud.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => choose(p.id)}
            className={cn(
              'flex h-16 flex-col items-start justify-center rounded-xl border px-3 text-left transition-colors',
              d.cloudProvider === p.id ? 'border-accent/60 bg-accent/[0.06]' : 'border-border bg-surface/70 hover:border-border-strong'
            )}
          >
            <span className="text-sm font-medium text-fg">{p.label}</span>
            <span className={cn('text-2xs', p.key_set ? 'text-success' : 'text-fg-subtle')}>{p.key_set ? 'Key saved' : 'Needs a key'}</span>
          </button>
        ))}
      </div>

      {selected && (
        <Card className="space-y-4 p-5">
          {selected.key_set ? (
            <div className="flex items-center gap-2 text-sm text-fg">
              <IconCircleCheck size={17} className="text-success" />
              Your {selected.label} key is saved in the system keychain.
            </div>
          ) : (
            <>
              {selected.id === 'openrouter' && <OpenRouterConnect />}
              {selected.id === 'anthropic' && <PlanSteps title="Have a Claude Max or Team plan? Use its included API credits" markdown={CLAUDE_PLAN_STEPS} />}
              {selected.id === 'nous' && <InstructionsGuide markdown={NOUS_STEPS} />}
              <Field
                label={selected.id === 'openrouter' ? 'Or paste an OpenRouter key' : `${selected.label} API key`}
                description="Stored in your operating system's keychain. Never written to files or logs."
              >
                <form
                  className="flex gap-2"
                  onSubmit={(e) => {
                    e.preventDefault()
                    if (!key.trim()) return
                    setSecret.mutate(
                      { name: selected.id, value: key.trim() },
                      {
                        onSuccess: () => {
                          setKey('')
                          toast.success('Key saved')
                        },
                        onError: (err) => toast.error("Couldn't save the key", { description: errorMessage(err) })
                      }
                    )
                  }}
                >
                  <Input
                    type={show ? 'text' : 'password'}
                    autoComplete="off"
                    value={key}
                    onChange={(e) => setKey(e.target.value)}
                    placeholder="Paste your API key"
                    leftIcon={<IconKey />}
                    className="font-mono text-xs"
                    rightSlot={<IconButton size="xs" tooltip={false} label={show ? 'Hide' : 'Show'} icon={show ? <IconEyeOff size={14} /> : <IconEye size={14} />} onClick={() => setShow((s) => !s)} />}
                  />
                  <Button type="submit" variant="primary" loading={setSecret.isPending} disabled={!key.trim()}>
                    Save
                  </Button>
                </form>
                <button type="button" onClick={() => void getBridge().openExternal(selected.docs_url)} className="mt-2 flex items-center gap-1 text-xs text-accent-text hover:underline">
                  Get a {selected.label} key <IconExternalLink size={12} />
                </button>
              </Field>
            </>
          )}
          <Field label="Main model">
            <div className="flex items-center gap-2">
              <ModelPicker provider={selected.id} value={d.primary} onChange={(primary) => d.set({ primary })} className="flex-1" />
              <ModelTest model={d.primary} role="primary" className="shrink-0" />
            </div>
          </Field>
          <Field label="Background model" description="Used for memory and summaries. A cheaper model is ideal.">
            <ModelPicker provider={selected.id} value={d.fast} onChange={(fast) => d.set({ fast })} />
          </Field>
          <Field label="Memory search" description="Embeddings can stay local even with a cloud chat model.">
            <div className="flex items-center gap-2">
              <ModelPicker embedding value={d.embedding} onChange={(embedding) => d.set({ embedding })} className="flex-1" />
              <ModelTest embedding model={d.embedding} className="shrink-0" />
            </div>
          </Field>
        </Card>
      )}
      {selected?.key_set && <DraftCheckup />}
    </div>
  )
}

/** Steps for using a plan, folded away until asked for. */
function PlanSteps({ title, markdown }: { title: string; markdown: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-xl border border-border bg-surface/60 px-3.5 py-2.5">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 text-left text-sm font-medium text-fg"
      >
        {title}
        <span className="shrink-0 text-xs font-normal text-accent-text">{open ? 'Hide steps' : 'Show steps'}</span>
      </button>
      {open && <InstructionsGuide markdown={markdown} className="mt-3" />}
    </div>
  )
}
