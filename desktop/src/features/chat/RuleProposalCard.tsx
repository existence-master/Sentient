/**
 * "Make this a rule?": the user said something like "never delete my emails" and Sentient proposes the matching
 * lasting rule (docs/API.md section 2, #130). Only a click here creates it. Until then the chat asks first.
 */
import { IconCheck, IconShieldLock } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { toast } from 'sonner'
import { Button } from '@/components/ui'
import { useDecideRuleProposal } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import type { RuleProposal } from '@/lib/types'
import { truncate } from '@/lib/utils'

export function RuleProposalCard({ proposal }: { proposal: RuleProposal }) {
  const decide = useDecideRuleProposal()
  const level = proposal.rule === 'never' ? 'Never' : 'Always ask'
  const what = proposal.targets.map((t) => t.label).join(', ')

  const answer = (decision: 'accept' | 'decline') =>
    decide.mutate(
      { id: proposal.id, decision, sessionId: proposal.session_id },
      {
        onSuccess: (p) => {
          if (p.status === 'accepted') toast.success('Rule saved', { description: 'You can change it in Settings > Approvals & safety.' })
        },
        onError: (err) => toast.error("Couldn't save your answer", { description: errorMessage(err) })
      }
    )

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="overflow-hidden rounded-xl border border-accent/30 bg-accent/[0.05] shadow-soft"
    >
      <div className="flex items-start gap-3 px-4 pt-3.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent-text">
          <IconShieldLock size={17} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-fg">Make this a rule?</div>
          <div className="mt-1 text-md font-medium text-fg">
            {level}: {what}
          </div>
          <p className="mt-0.5 text-sm text-fg-muted">
            You said “{truncate(proposal.said, 160)}”. A rule keeps working in every chat, even long after this one.
          </p>
          <p className="mt-1 text-xs text-fg-subtle">Until you choose, Sentient checks with you before using it in this chat.</p>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2 px-4 py-3">
        <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} loading={decide.isPending} onClick={() => answer('accept')}>
          Make it a rule
        </Button>
        <Button size="sm" variant="ghost" disabled={decide.isPending} onClick={() => answer('decline')}>
          Not now
        </Button>
      </div>
    </motion.div>
  )
}
