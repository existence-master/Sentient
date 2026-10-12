/**
 * The one-time "You're using your ChatGPT plan" message, shown after the first Sign in with ChatGPT on this computer
 * (OpenAI's UI guidelines: https://developers.openai.com/siwc/ui-ux-guidelines). The engine remembers it was shown
 * (`welcome` in `GET /api/models/connect/chatgpt`), so it never comes back, even after signing in again.
 */
import { Button, Dialog } from '@/components/ui'
import { openExternal } from '@/features/integrations/InstructionsGuide'
import { useChatGPTStatus, useChatGPTWelcomed } from '@/hooks/models'
import { ChatGPTLogo } from './ChatGPTBrand'

export function ChatGPTWelcome() {
  const status = useChatGPTStatus()
  const welcomed = useChatGPTWelcomed()
  const open = !!status.data?.signed_in && !!status.data.welcome && welcomed.isIdle
  const manageUsage = status.data?.manage_usage_url ?? 'https://chatgpt.com/settings/usage'
  const dismiss = () => welcomed.mutate()

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && dismiss()}
      size="sm"
      hideClose
      title={<span className="sr-only">You&apos;re using your ChatGPT plan</span>}
      footer={
        <Button autoFocus variant="primary" onClick={dismiss}>
          Got it
        </Button>
      }
    >
      <div className="flex flex-col items-center px-2 pb-1 text-center">
        <div className="mb-4 flex size-12 items-center justify-center rounded-2xl bg-black dark:bg-white">
          <ChatGPTLogo tone="white" size={28} className="dark:hidden" />
          <ChatGPTLogo tone="black" size={28} className="hidden dark:block" />
        </div>
        <h2 aria-hidden className="text-md font-semibold text-fg">
          You&apos;re using your ChatGPT plan
        </h2>
        <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">
          Eligible usage in this app uses your ChatGPT plan. Manage usage in your{' '}
          <button type="button" className="text-accent-text hover:underline" onClick={() => openExternal(manageUsage)}>
            ChatGPT settings
          </button>
          .
        </p>
      </div>
    </Dialog>
  )
}
