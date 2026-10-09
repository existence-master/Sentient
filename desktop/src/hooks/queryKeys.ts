/**
 * Every React Query key in one place. Invalidate by prefix, e.g.
 * `queryClient.invalidateQueries({ queryKey: qk.tasks.all })`.
 */
import type { MemoryQuery } from '@/lib/types'

export const qk = {
  bootstrap: ['bootstrap'] as const,
  stop: ['stop'] as const,
  config: ['config'] as const,
  configSchema: ['config-schema'] as const,

  sessions: ['sessions'] as const,
  /** Kept outside `sessions` so invalidating the list doesn't refetch transcripts mid-stream. */
  messages: (sessionId: string) => ['messages', sessionId] as const,
  sessionSearch: (q: string) => ['session-search', q] as const,

  providers: ['models', 'providers'] as const,
  localModels: ['models', 'local'] as const,
  modelPresets: ['models', 'presets'] as const,
  secrets: ['secrets'] as const,

  tasks: {
    all: ['tasks'] as const,
    detail: (id: string) => ['tasks', id] as const,
    runEvents: (id: string, runId: string) => ['tasks', id, 'runs', runId, 'events'] as const
  },

  integrations: {
    all: ['integrations'] as const,
    detail: (id: string) => ['integrations', id] as const,
    privacyFilters: (id: string) => ['integrations', id, 'privacy-filters'] as const,
    mcp: ['integrations-mcp'] as const,
    feeds: ['integrations', 'feeds'] as const
  },
  hooks: ['hooks'] as const,

  notifications: ['notifications'] as const,
  proactivity: {
    status: ['proactivity', 'status'] as const,
    preferences: ['proactivity', 'preferences'] as const,
    brief: ['proactivity', 'brief'] as const
  },

  memories: {
    all: ['memories'] as const,
    list: (q: MemoryQuery) => ['memories', 'list', q] as const,
    topics: ['memories', 'topics'] as const,
    graph: ['memories', 'graph'] as const,
    summaries: ['memories', 'summaries'] as const,
    workspace: ['memories', 'workspace'] as const,
    personas: ['memories', 'personas'] as const,
    dreams: ['memories', 'dreams'] as const
  },
  userModel: ['user-model'] as const,

  skills: {
    all: ['skills'] as const,
    detail: (name: string) => ['skills', 'detail', name] as const,
    diff: (name: string) => ['skills', 'diff', name] as const,
    evolutionLog: ['skills', 'evolution-log'] as const
  },

  voiceStatus: ['voice', 'status'] as const,
  sandboxStatus: ['sandbox', 'status'] as const,
  usage: (days: number) => ['usage', days] as const,
  files: ['files'] as const,
  tools: ['tools'] as const
}
