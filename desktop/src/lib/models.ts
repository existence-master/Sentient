/** Helpers for presenting and choosing LiteLLM model strings (`provider/model`). */
import type { LocalModel, LocalModels, Provider, RoleName } from './types'

export interface RoleMeta {
  role: RoleName
  label: string
  description: string
  required: boolean
  embedding?: boolean
  /** Which role an empty value falls back to. */
  fallsBackTo?: RoleName
}

export const ROLE_META: Record<RoleName, RoleMeta> = {
  primary: {
    role: 'primary',
    label: 'Primary',
    description: 'Your main conversational model. It talks to you, uses tools and makes decisions.',
    required: true
  },
  fast: {
    role: 'fast',
    label: 'Fast / background',
    description: 'Cheap and quick. Learns facts about you, writes summaries, names chats and triages events.',
    required: true
  },
  voice: {
    role: 'voice',
    label: 'Voice',
    description: 'Spoken conversations, so pick something fast. Leave empty to use the primary model.',
    required: false,
    fallsBackTo: 'primary'
  },
  planner: {
    role: 'planner',
    label: 'Planner',
    description: 'Turns requests into task plans. Leave empty to use the primary model.',
    required: false,
    fallsBackTo: 'primary'
  },
  executor: {
    role: 'executor',
    label: 'Executor',
    description: 'Carries out long-running tasks step by step. Leave empty to use the primary model.',
    required: false,
    fallsBackTo: 'primary'
  },
  embedding: {
    role: 'embedding',
    label: 'Embedding',
    description: 'Turns memories into vectors so Sentient can find relevant ones. Changing it re-indexes memory.',
    required: true,
    embedding: true
  },
  vision: {
    role: 'vision',
    label: 'Vision',
    description: 'Looks at images you attach. Leave empty to use the primary model.',
    required: false,
    fallsBackTo: 'primary'
  }
}

export const REASONING_LEVELS = ['none', 'low', 'medium', 'high'] as const

const PROVIDER_LABELS: Record<string, string> = {
  ollama_chat: 'Ollama',
  ollama: 'Ollama',
  lm_studio: 'LM Studio',
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  gemini: 'Gemini',
  openrouter: 'OpenRouter',
  groq: 'Groq',
  mistral: 'Mistral',
  deepseek: 'DeepSeek',
  xai: 'xAI'
}

export function providerOf(model: string | null | undefined): string {
  if (!model) return ''
  const i = model.indexOf('/')
  return i > 0 ? model.slice(0, i) : ''
}

export function providerLabel(id: string): string {
  return PROVIDER_LABELS[id] ?? id
}

/** `ollama_chat/qwen3:8b` -> `qwen3:8b`; `openrouter/anthropic/claude` -> `anthropic/claude`. */
export function modelShortName(model: string | null | undefined): string {
  if (!model) return ''
  const i = model.indexOf('/')
  return i > 0 ? model.slice(i + 1) : model
}

export function isLocalModel(model: string | null | undefined): boolean {
  return ['ollama', 'ollama_chat', 'lm_studio'].includes(providerOf(model))
}

export function looksLikeEmbedding(name: string): boolean {
  return /embed|bge|minilm|e5-|gte-|nomic/i.test(name)
}

export interface ModelOption {
  value: string
  label: string
  hint?: string
  badge?: string
  disabled?: boolean
}

export interface ModelOptionGroup {
  id: string
  label: string
  options: ModelOption[]
  note?: string
}

export function localModelValue(runtime: 'ollama' | 'lm_studio', m: LocalModel, embedding: boolean): string {
  if (runtime === 'lm_studio') return `lm_studio/${m.name}`
  return `${embedding ? 'ollama' : 'ollama_chat'}/${m.name}`
}

/** Groups for the model combobox: detected local models first, then provider suggestions. */
export function buildModelOptions(
  local: LocalModels | undefined,
  providers: Provider[] | undefined,
  opts: { embedding?: boolean } = {}
): ModelOptionGroup[] {
  const embedding = !!opts.embedding
  const groups: ModelOptionGroup[] = []
  const wantLocal = (m: LocalModel) => (m.is_embedding ?? looksLikeEmbedding(m.name)) === embedding

  if (local?.ollama.reachable) {
    const options = local.ollama.models.filter(wantLocal).map((m) => ({
      value: localModelValue('ollama', m, embedding),
      label: m.name,
      hint: [m.parameter_size, m.family].filter(Boolean).join(' · ')
    }))
    if (options.length) groups.push({ id: 'ollama', label: 'Installed in Ollama', options })
  }
  if (local?.lm_studio.reachable && local.lm_studio.models.length) {
    groups.push({
      id: 'lm_studio',
      label: 'LM Studio',
      options: local.lm_studio.models
        .filter((m) => looksLikeEmbedding(m.name) === embedding)
        .map((m) => ({ value: localModelValue('lm_studio', m, embedding), label: m.name }))
    })
  }
  for (const p of providers ?? []) {
    if (p.kind === 'local') continue
    const options = p.suggested
      .filter((s) => looksLikeEmbedding(s) === embedding)
      .map((s) => ({
        value: s,
        label: modelShortName(s),
        badge: p.key_required && !p.key_set ? 'needs key' : undefined
      }))
    if (options.length) groups.push({ id: p.id, label: p.label, options })
  }
  return groups.filter((g) => g.options.length)
}

const CHAT_PREFERENCE = [/^qwen3:8b/, /^qwen3:14b/, /^qwen3/, /^llama3\.1:8b/, /^qwen2\.5:7b/, /^gpt-oss/, /^llama/, /^qwen/, /^mistral/]
const EMBED_PREFERENCE = [/^nomic-embed-text/, /^mxbai-embed/, /embed/]

export function recommendLocal(local: LocalModels | undefined, embedding = false): string | null {
  if (!local?.ollama.reachable) return null
  const pool = local.ollama.models.filter((m) => (m.is_embedding ?? looksLikeEmbedding(m.name)) === embedding)
  for (const re of embedding ? EMBED_PREFERENCE : CHAT_PREFERENCE) {
    const hit = pool.find((m) => re.test(m.name))
    if (hit) return localModelValue('ollama', hit, embedding)
  }
  return pool[0] ? localModelValue('ollama', pool[0], embedding) : null
}

/** Smallest good chat model for the fast role. */
export function recommendFastLocal(local: LocalModels | undefined): string | null {
  if (!local?.ollama.reachable) return null
  const pool = local.ollama.models.filter((m) => !(m.is_embedding ?? looksLikeEmbedding(m.name)))
  const hit = pool.find((m) => /^qwen3:4b/.test(m.name)) ?? pool.find((m) => /^qwen3:8b/.test(m.name))
  return hit ? localModelValue('ollama', hit, false) : recommendLocal(local)
}

export const SUGGESTED_PULLS = [
  { name: 'qwen3:8b', size: '5.2 GB', note: 'Best all-rounder with tool use' },
  { name: 'qwen3:4b', size: '2.5 GB', note: 'Lighter, good for the fast role' },
  { name: 'nomic-embed-text', size: '274 MB', note: 'Embeddings for memory' }
]
