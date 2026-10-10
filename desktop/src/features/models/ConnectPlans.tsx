/**
 * Use the AI plans people already pay for (docs/API.md §3, "Connecting plans"):
 * a Claude Max or Team plan's API credits, Sign in with ChatGPT for a ChatGPT plan, OpenRouter's browser sign-in and
 * a Nous Portal key. Every key and sign-in goes to the system keychain; removing it disconnects.
 */
import {
  IconCircleCheck,
  IconExternalLink,
  IconEye,
  IconEyeOff,
  IconInfoCircle,
  IconKey,
  IconLogin,
  IconLogout,
  IconPlugConnected,
  IconTrash
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, IconButton, Input, SegmentedControl } from '@/components/ui'
import { InstructionsGuide, openExternal } from '@/features/integrations/InstructionsGuide'
import { usePresetSwitch } from '@/features/models/usePresetSwitch'
import {
  useChatGPTStatus,
  useCheckKey,
  useDeleteSecret,
  useModelCatalog,
  useModelPresets,
  useProviders,
  useSecretSaves,
  useSetRoles,
  useSetSecret,
  useSignInStatus,
  useSignOutChatGPT
} from '@/hooks/models'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { looksLikeEmbedding } from '@/lib/models'
import type { ProviderKeyCheck } from '@/lib/types'

export const CLAUDE_PLAN_STEPS = `Claude Max and Team plans come with monthly API credits ($100 a month on Max 5x, $200 on Max 20x). Sentient uses them through an ordinary API key, so your usage limits in the Claude app stay the same.

1. On claude.ai, open **Settings > Billing** (on a Team plan, **Organization settings > Billing**): https://claude.ai/settings/billing
2. Under **API credits**, choose the Claude Console organization that should get the credits and link it. You can't change this yourself later, so pick the one you'll keep using.
3. In the Claude Console, create an API key: https://platform.claude.com/settings/keys
4. Paste the key here and test it.`

export const NOUS_STEPS = `Nous Portal doesn't offer a sign-in for other apps yet, so Sentient uses an API key.

1. Open Nous Portal and sign in: https://portal.nousresearch.com
2. Open the API keys page and create a new key.
3. Paste the key here and test it. Its models then show up in the model lists.`

/** Paste a key straight into the keychain. */
function KeyPaste({ provider, label, placeholder }: { provider: string; label: string; placeholder: string }) {
  const [value, setValue] = useState('')
  const [show, setShow] = useState(false)
  const save = useSetSecret()
  const submit = () => {
    if (!value.trim()) return
    save.mutate(
      { name: provider, value: value.trim() },
      {
        onSuccess: () => {
          setValue('')
          toast.success(`${label} key saved`, {
            description: 'Stored in your system keychain.'
          })
        },
        onError: (e) =>
          toast.error("Couldn't save the key", {
            description: errorMessage(e)
          })
      }
    )
  }
  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <Input
        type={show ? 'text' : 'password'}
        autoComplete="off"
        aria-label={`${label} API key`}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
        leftIcon={<IconKey />}
        className="font-mono text-xs"
        wrapperClassName="flex-1"
        rightSlot={
          <IconButton
            size="xs"
            tooltip={false}
            label={show ? 'Hide key' : 'Show key'}
            icon={show ? <IconEyeOff size={14} /> : <IconEye size={14} />}
            onClick={() => setShow((s) => !s)}
          />
        }
      />
      <Button type="submit" variant="primary" loading={save.isPending} disabled={!value.trim()}>
        Save
      </Button>
    </form>
  )
}

/** "Key saved" with Test and Remove, or the paste field when there is no key yet. */
function KeyStatus({
  provider,
  label,
  placeholder,
  removeLabel = 'Remove key',
  onChecked
}: {
  provider: string
  label: string
  placeholder: string
  removeLabel?: string
  onChecked?: (result: ProviderKeyCheck) => void
}) {
  const providers = useProviders()
  const check = useCheckKey()
  const remove = useDeleteSecret()
  const keySet = !!providers.data?.find((p) => p.id === provider)?.key_set
  const result = check.data

  if (!keySet) return <KeyPaste provider={provider} label={label} placeholder={placeholder} />
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 text-sm text-fg">
          <IconCircleCheck size={16} className="text-success" /> {label} is set up. The key is in your system keychain.
        </span>
        <div className="flex-1" />
        <Button size="sm" variant="secondary" loading={check.isPending} onClick={() => check.mutate(provider, { onSuccess: onChecked })}>
          Test
        </Button>
        <Button
          size="sm"
          variant="ghost"
          leftIcon={<IconTrash size={13} />}
          loading={remove.isPending}
          onClick={() =>
            remove.mutate(provider, {
              onSuccess: () => {
                check.reset()
                toast.success(`${label} key removed`)
              },
              onError: (e) =>
                toast.error("Couldn't remove the key", {
                  description: errorMessage(e)
                })
            })
          }
        >
          {removeLabel}
        </Button>
      </div>
      {result && (
        <Alert tone={result.ok ? 'success' : 'danger'} className="py-2">
          {result.ok ? result.detail : result.error}
        </Alert>
      )}
    </div>
  )
}

/** The Claude Max or Team route: claim the included API credits, paste the key, test it, use Claude for the main roles. */
export function ClaudePlanGuide({ onUseModels }: { onUseModels?: (primary: string, fast: string) => void }) {
  const providers = useProviders()
  const [works, setWorks] = useState(false)
  const anthropic = providers.data?.find((p) => p.id === 'anthropic')
  const keySet = !!anthropic?.key_set
  const suggested = (anthropic?.suggested ?? []).filter((m) => !looksLikeEmbedding(m))
  const primary = suggested[0]
  const fast = suggested.find((m) => /haiku/i.test(m)) ?? primary

  // A removed or replaced key needs a new test before Claude can be picked for the roles.
  const saves = useSecretSaves('anthropic')
  useEffect(() => {
    setWorks(false)
  }, [keySet, saves])

  return (
    <div className="space-y-4">
      <InstructionsGuide markdown={CLAUDE_PLAN_STEPS} />
      <KeyStatus provider="anthropic" label="Claude" placeholder="sk-ant-…" onChecked={(r) => setWorks(r.ok)} />
      {keySet && works && primary && onUseModels && (
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant="primary" onClick={() => onUseModels(primary, fast)}>
            Use Claude for every job
          </Button>
          <span className="text-xs text-fg-subtle">You can still pick a model per job under Roles.</span>
        </div>
      )}
      <Alert tone="info" icon={<IconInfoCircle />}>
        Pro and Free plans don't include API credits, so a key bills your Claude Console account per use. Apps aren't allowed to sign in with your Claude
        account, which is why Sentient asks for a key. Credits arrive each billing month, don't roll over and don't cover Claude Code. When they run out, Claude
        stops answering until the next month unless you add credits in the Console.
      </Alert>
    </div>
  )
}

/** One click: OpenRouter's own sign-in in the browser hands Sentient a key. */
export function OpenRouterConnect() {
  const providers = useProviders()
  const remove = useDeleteSecret()
  const check = useCheckKey()
  const [flow, setFlow] = useState<{ state: string; url: string } | null>(null)
  const [starting, setStarting] = useState(false)
  const connected = !!providers.data?.find((p) => p.id === 'openrouter')?.key_set
  const catalog = useModelCatalog('openrouter', connected)
  const status = useSignInStatus(flow?.state ?? null)

  useEffect(() => {
    if (status.data?.status === 'connected') {
      toast.success('OpenRouter is connected', {
        description: 'The key is in your system keychain.'
      })
      setFlow(null)
    }
  }, [status.data?.status])

  const start = async () => {
    setStarting(true)
    try {
      const res = await api.models.connectOpenRouter()
      setFlow({ state: res.state, url: res.auth_url })
      openExternal(res.auth_url)
    } catch (e) {
      toast.error("Couldn't start the sign-in", {
        description: errorMessage(e)
      })
    } finally {
      setStarting(false)
    }
  }

  const free = catalog.data?.filter((m) => m.free).length ?? 0
  return (
    <div className="space-y-3">
      <p className="text-sm leading-relaxed text-fg-muted">
        OpenRouter gives you hundreds of models with one account. Your browser opens OpenRouter; sign in and approve, and Sentient gets its own key, kept in
        your system keychain. You pay OpenRouter per use from your credits. Models marked Free cost nothing but have daily limits. You can set a spending limit
        for Sentient's key on OpenRouter's Keys page.
      </p>
      {connected ? (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="success" size="sm">
              Connected
            </Badge>
            {catalog.data && (
              <span className="text-xs text-fg-subtle">
                {catalog.data.length} models, {free} free. Pick them under Roles.
              </span>
            )}
            <div className="flex-1" />
            <Button size="sm" variant="secondary" loading={check.isPending} onClick={() => check.mutate('openrouter')}>
              Test
            </Button>
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<IconTrash size={13} />}
              loading={remove.isPending}
              onClick={() =>
                remove.mutate('openrouter', {
                  onSuccess: () => {
                    check.reset()
                    toast.success('OpenRouter disconnected', {
                      description: 'The key was removed from your keychain.'
                    })
                  },
                  onError: (e) =>
                    toast.error("Couldn't disconnect", {
                      description: errorMessage(e)
                    })
                })
              }
            >
              Disconnect
            </Button>
          </div>
          {check.data && (
            <Alert tone={check.data.ok ? 'success' : 'danger'} className="py-2">
              {check.data.ok ? check.data.detail : check.data.error}
            </Alert>
          )}
        </div>
      ) : flow ? (
        <div className="space-y-2">
          {status.data?.status === 'failed' ? (
            <Alert
              tone="danger"
              title="OpenRouter didn't connect"
              action={
                <Button size="xs" onClick={() => void start()}>
                  Try again
                </Button>
              }
            >
              {status.data.error}
            </Alert>
          ) : (
            <div className="flex flex-wrap items-center gap-2 text-sm text-fg-muted">
              <span className="flex-1">Waiting for you to approve in your browser…</span>
              <Button size="sm" variant="secondary" onClick={() => openExternal(flow.url)}>
                Open the page again
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setFlow(null)}>
                Cancel
              </Button>
            </div>
          )}
        </div>
      ) : (
        <Button variant="primary" leftIcon={<IconLogin size={15} />} loading={starting} onClick={() => void start()}>
          Connect OpenRouter
        </Button>
      )}
    </div>
  )
}

/** Sign in with ChatGPT: a ChatGPT Plus or Pro plan's models for any job, through OpenAI's sign-in for open-source apps. */
export function ChatGPTConnect({ onUseModels }: { onUseModels?: (primary: string, fast: string) => void }) {
  const qc = useQueryClient()
  const status = useChatGPTStatus()
  const signOut = useSignOutChatGPT()
  const check = useCheckKey()
  const [flow, setFlow] = useState<{ state: string; url: string } | null>(null)
  const [starting, setStarting] = useState(false)
  const signedIn = !!status.data?.signed_in
  const catalog = useModelCatalog('chatgpt', signedIn)
  const flowStatus = useSignInStatus(flow?.state ?? null, 'chatgpt')
  const manageUsage = status.data?.manage_usage_url ?? 'https://chatgpt.com/settings/usage'

  useEffect(() => {
    if (flowStatus.data?.status === 'connected') {
      toast.success("You're using your ChatGPT plan", {
        description: 'Your sign-in is kept in your system keychain. Pick ChatGPT models under Roles.'
      })
      setFlow(null)
      void qc.invalidateQueries({ queryKey: qk.chatgpt })
    }
  }, [flowStatus.data?.status, qc])

  const start = async () => {
    setStarting(true)
    try {
      const res = await api.models.chatgpt.connect()
      setFlow({ state: res.state, url: res.auth_url })
      openExternal(res.auth_url)
    } catch (e) {
      toast.error("Couldn't start the sign-in", {
        description: errorMessage(e)
      })
    } finally {
      setStarting(false)
    }
  }

  const models = (catalog.data ?? []).map((m) => m.id).filter((m) => !looksLikeEmbedding(m))
  const main = models[0]
  const fast = models.find((m) => /mini|nano/i.test(m)) ?? main

  if (status.isError) {
    return <Alert tone="danger">{errorMessage(status.error)}</Alert>
  }
  if (status.data && !status.data.available) {
    return (
      <Alert tone="info" icon={<IconInfoCircle />} title="Sign in with ChatGPT isn't available yet">
        {status.data.reason}
      </Alert>
    )
  }
  return (
    <div className="space-y-3">
      <p className="text-sm leading-relaxed text-fg-muted">
        Use your ChatGPT Plus or Pro plan for chat and tasks. Your browser opens ChatGPT; sign in and allow Sentient. Usage counts against your plan, not
        an API bill, and you can set a limit for Sentient in ChatGPT's settings. Your sign-in stays in your system keychain and renews on its own. ChatGPT
        plans don't include models for memory search, so that keeps its current model.
      </p>
      {signedIn ? (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="success" size="sm">
              Using your ChatGPT plan
            </Badge>
            <span className="text-xs text-fg-subtle">
              {status.data?.email ? `${status.data.email}. ` : ''}
              {catalog.data ? `${catalog.data.length} models on your plan. Pick them under Roles.` : catalog.isError ? errorMessage(catalog.error) : ''}
            </span>
            <div className="flex-1" />
            <Button size="sm" variant="secondary" loading={check.isPending} onClick={() => check.mutate('chatgpt')}>
              Test
            </Button>
            <Button size="sm" variant="ghost" leftIcon={<IconExternalLink size={13} />} onClick={() => openExternal(manageUsage)}>
              Manage usage
            </Button>
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<IconLogout size={13} />}
              loading={signOut.isPending}
              onClick={() =>
                signOut.mutate(undefined, {
                  onSuccess: () => {
                    check.reset()
                    toast.success('Signed out of ChatGPT', {
                      description: 'The sign-in was removed from your keychain.'
                    })
                  },
                  onError: (e) =>
                    toast.error("Couldn't sign out", {
                      description: errorMessage(e)
                    })
                })
              }
            >
              Sign out
            </Button>
          </div>
          {check.data && (
            <Alert tone={check.data.ok ? 'success' : 'danger'} className="py-2">
              {check.data.ok ? check.data.detail : check.data.error}
            </Alert>
          )}
          {main && onUseModels && (
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="primary" onClick={() => onUseModels(main, fast ?? main)}>
                Use ChatGPT for every job
              </Button>
              <span className="text-xs text-fg-subtle">You can still pick a model per job under Roles.</span>
            </div>
          )}
        </div>
      ) : flow ? (
        <div className="space-y-2">
          {flowStatus.data?.status === 'failed' ? (
            <Alert
              tone="danger"
              title="ChatGPT didn't connect"
              action={
                <Button size="xs" onClick={() => void start()}>
                  Try again
                </Button>
              }
            >
              {flowStatus.data.error}
            </Alert>
          ) : (
            <div className="flex flex-wrap items-center gap-2 text-sm text-fg-muted">
              <span className="flex-1">Waiting for you to sign in to ChatGPT in your browser…</span>
              <Button size="sm" variant="secondary" onClick={() => openExternal(flow.url)}>
                Open the page again
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setFlow(null)}>
                Cancel
              </Button>
            </div>
          )}
        </div>
      ) : (
        <Button variant="primary" leftIcon={<IconLogin size={15} />} loading={starting || status.isLoading} onClick={() => void start()}>
          Continue with ChatGPT
        </Button>
      )}
    </div>
  )
}

export function NousPortalGuide() {
  return (
    <div className="space-y-4">
      <InstructionsGuide markdown={NOUS_STEPS} />
      <KeyStatus provider="nous" label="Nous Portal" placeholder="Paste your Nous Portal key" />
      <p className="text-xs leading-relaxed text-fg-subtle">Usage counts against your Nous Portal credits or subscription.</p>
    </div>
  )
}

type Plan = 'claude' | 'chatgpt' | 'openrouter' | 'nous'

/** Settings > Models: "Use a plan you already have". */
export function ConnectPlansSection() {
  const [plan, setPlan] = useState<Plan>('claude')
  const setRoles = useSetRoles()
  const presets = useModelPresets()
  const presetSwitch = usePresetSwitch()
  // When the built-in Cloud preset uses this provider, switch to it so it can be undone; otherwise set the two roles.
  const applyPlan = (provider: string, label: string) => (primary: string, fast: string) => {
    const preset = presets.data?.presets.find((p) => p.builtin && p.available && p.provider === provider)
    if (preset) return presetSwitch.apply(preset.name)
    setRoles.mutate(
      { primary, fast },
      {
        onSuccess: () =>
          toast.success(`Sentient now uses ${label}`, {
            description: 'For chat and for background work.'
          }),
        onError: (e) =>
          toast.error("Couldn't change the models", {
            description: errorMessage(e)
          })
      }
    )
  }
  return (
    <section className="space-y-3">
      <div className="px-1">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-fg">
          <IconPlugConnected size={15} /> Use a plan you already have
        </h3>
        <p className="mt-0.5 text-xs text-fg-subtle">
          Claude Max credits, your ChatGPT plan, an OpenRouter account or Nous Portal. Keys and sign-ins stay in your system keychain.
        </p>
      </div>
      <Card className="space-y-4 p-4">
        <SegmentedControl
          value={plan}
          onChange={setPlan}
          options={[
            { value: 'claude', label: 'Claude plan' },
            { value: 'chatgpt', label: 'ChatGPT plan' },
            { value: 'openrouter', label: 'OpenRouter' },
            { value: 'nous', label: 'Nous Portal' }
          ]}
        />
        {plan === 'claude' ? (
          <ClaudePlanGuide onUseModels={applyPlan('anthropic', 'Claude')} />
        ) : plan === 'chatgpt' ? (
          <ChatGPTConnect onUseModels={applyPlan('chatgpt', 'your ChatGPT plan')} />
        ) : plan === 'openrouter' ? (
          <OpenRouterConnect />
        ) : (
          <NousPortalGuide />
        )}
      </Card>
    </section>
  )
}
