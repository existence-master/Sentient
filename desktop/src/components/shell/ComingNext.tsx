import type { Icon } from '@tabler/icons-react'
import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { Badge, PageHeader } from '@/components/ui'

export interface ComingNextFeature {
  icon: Icon
  title: string
  description: string
}

/**
 * Polished placeholder for screens another engineer is building.
 * Replace the page's body; keep the PageHeader pattern.
 */
export function ComingNext({
  icon: IconCmp,
  title,
  description,
  headline,
  features,
  children,
  actions
}: {
  icon: Icon
  title: string
  description: string
  headline: string
  features: ComingNextFeature[]
  children?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="h-full overflow-y-auto">
      <PageHeader icon={<IconCmp />} title={title} description={description} actions={actions} />
      <div className="max-w-4xl space-y-6 px-8 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-elevated/60 p-7"
        >
          <div
            aria-hidden
            className="pointer-events-none absolute -right-24 -top-24 size-72 rounded-full opacity-[0.12] blur-3xl"
            style={{ background: 'radial-gradient(circle, var(--accent), transparent 70%)' }}
          />
          <Badge tone="accent">Coming next</Badge>
          <h2 className="mt-3 max-w-xl text-lg font-semibold tracking-tight text-fg">{headline}</h2>
          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 * i }}
                className="flex gap-3 rounded-xl border border-border bg-surface p-3.5"
              >
                <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
                  <f.icon size={17} />
                </div>
                <div>
                  <div className="text-sm font-medium text-fg">{f.title}</div>
                  <div className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{f.description}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
        {children}
      </div>
    </div>
  )
}
