import { IconExternalLink } from '@tabler/icons-react'
import { Tooltip } from '@/components/ui'
import { openExternal } from '@/features/integrations/InstructionsGuide'
import { ChatGPTMark } from '@/features/models/ChatGPTBrand'
import { useConfig } from '@/hooks/core'
import { cn } from '@/lib/utils'

const MANAGE_USAGE = 'https://chatgpt.com/settings/usage'

/**
 * "Using ChatGPT plan" next to the model picker while the next message goes to a ChatGPT plan model, with the
 * Manage usage link (OpenAI's UI guidelines: https://developers.openai.com/siwc/ui-ux-guidelines).
 * `override` is the chat's own model (#312); without one the default model answers.
 */
export function PlanIndicator({ override, className }: { override: string | undefined; className?: string }) {
  const config = useConfig()
  const model = override ?? config.data?.models.roles.primary ?? ''
  if (!model.startsWith('chatgpt/')) return null
  return (
    <span className={cn('flex h-8 shrink-0 items-center gap-1.5 rounded-lg px-2 text-xs text-fg-subtle', className)}>
      <ChatGPTMark size={14} />
      <span className="font-medium text-fg-muted">Using ChatGPT plan</span>
      <Tooltip content="See or change what Sentient may use in ChatGPT's settings">
        <button
          type="button"
          onClick={() => openExternal(MANAGE_USAGE)}
          className="ml-0.5 flex items-center gap-0.5 rounded px-1 text-accent-text hover:bg-hover hover:underline"
        >
          Manage usage <IconExternalLink size={11} />
        </button>
      </Tooltip>
    </span>
  )
}
