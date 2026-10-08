/** Skills presentation helpers: authors, evolution kinds, SKILL.md parsing and a line diff. */
import {
  IconArchive,
  IconBrush,
  IconFilePencil,
  IconFirstAidKit,
  IconMessages,
  IconSparkles,
  IconUser,
  IconUserCog,
  IconWorld,
  type Icon
} from '@tabler/icons-react'
import type { Skill } from '@/lib/types'

export const AUTHOR_META: Record<string, { label: string; icon: Icon; description: string }> = {
  user: { label: 'You', icon: IconUser, description: 'Written by you' },
  assistant: { label: 'Sentient', icon: IconSparkles, description: 'Learned by Sentient from your work' },
  community: { label: 'Community', icon: IconWorld, description: 'Shared by the community' }
}

export function authorMeta(author: string) {
  return AUTHOR_META[author] ?? { label: author, icon: IconUser, description: author }
}

export const EVOLUTION_META: Record<string, { label: string; icon: Icon; color: string }> = {
  skill_created: { label: 'Skill created', icon: IconSparkles, color: 'var(--accent)' },
  skill_patched: { label: 'Skill improved', icon: IconFilePencil, color: 'var(--info)' },
  skill_repair_proposed: { label: 'Fix proposed', icon: IconFirstAidKit, color: 'var(--warning)' },
  skill_archived: { label: 'Skill archived', icon: IconArchive, color: 'var(--fg-subtle)' },
  profile_updated: { label: 'Profile updated', icon: IconUserCog, color: 'var(--success)' },
  curator_run: { label: 'Curator tidied up', icon: IconBrush, color: 'var(--warning)' },
  summary_created: { label: 'Conversation remembered', icon: IconMessages, color: '#a78bfa' }
}

export function evolutionMeta(kind: string) {
  return EVOLUTION_META[kind] ?? { label: kind.replace(/_/g, ' '), icon: IconSparkles, color: 'var(--fg-subtle)' }
}

/** Strip a leading YAML frontmatter block. */
export function splitFrontmatter(md: string): { front: string; body: string } {
  const m = /^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/.exec(md)
  return m ? { front: m[1], body: m[2].trim() } : { front: '', body: md.trim() }
}

export interface SkillSection {
  title: string
  content: string
}

/** Split a SKILL.md body into its `##` sections (text before the first one is the intro). */
export function parseSections(body: string): { title: string; intro: string; sections: SkillSection[] } {
  const lines = body.split(/\r?\n/)
  let title = ''
  const intro: string[] = []
  const sections: SkillSection[] = []
  for (const line of lines) {
    const h1 = /^#\s+(.+)$/.exec(line)
    const h2 = /^##\s+(.+)$/.exec(line)
    if (h1 && !title && !sections.length) title = h1[1].trim()
    else if (h2) sections.push({ title: h2[1].trim(), content: '' })
    else if (sections.length) sections[sections.length - 1].content += `${line}\n`
    else intro.push(line)
  }
  return { title, intro: intro.join('\n').trim(), sections: sections.map((s) => ({ ...s, content: s.content.trim() })) }
}

export const SKILL_TEMPLATE = `# New skill

## When to use
Describe the situation or request that should trigger this skill.

## Procedure
1. First step
2. Second step

## Pitfalls
- What to avoid

## Verification
- How to tell it worked
`

export function slugify(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
}

export const validSkillName = (name: string) => /^[a-z0-9][a-z0-9-]{1,63}$/.test(name) && !['pending', 'archived'].includes(name)

// ---------------------------------------------------------------------------- diff
export type DiffRow =
  | { type: 'same'; left: number; right: number; text: string }
  | { type: 'del'; left: number; text: string }
  | { type: 'add'; right: number; text: string }
  | { type: 'change'; left: number; right: number; from: string; to: string }

/** Line-level LCS diff; adjacent delete/add runs are paired into `change` rows for side-by-side display. */
export function diffLines(a: string, b: string): DiffRow[] {
  const A = a.replace(/\r\n/g, '\n').split('\n')
  const B = b.replace(/\r\n/g, '\n').split('\n')
  const n = A.length
  const m = B.length
  const dp: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1))
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
  }
  const raw: DiffRow[] = []
  let i = 0
  let j = 0
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) {
      raw.push({ type: 'same', left: i + 1, right: j + 1, text: A[i] })
      i++
      j++
    } else if (j < m && (i >= n || dp[i][j + 1] >= dp[i + 1][j])) {
      raw.push({ type: 'add', right: j + 1, text: B[j] })
      j++
    } else {
      raw.push({ type: 'del', left: i + 1, text: A[i] })
      i++
    }
  }
  // pair runs of deletions with following additions
  const out: DiffRow[] = []
  for (let k = 0; k < raw.length; ) {
    if (raw[k].type !== 'del' && raw[k].type !== 'add') {
      out.push(raw[k++])
      continue
    }
    const dels: Array<Extract<DiffRow, { type: 'del' }>> = []
    const adds: Array<Extract<DiffRow, { type: 'add' }>> = []
    while (k < raw.length && (raw[k].type === 'del' || raw[k].type === 'add')) {
      const r = raw[k++]
      if (r.type === 'del') dels.push(r)
      else if (r.type === 'add') adds.push(r)
    }
    const pairs = Math.min(dels.length, adds.length)
    for (let p = 0; p < pairs; p++) out.push({ type: 'change', left: dels[p].left, right: adds[p].right, from: dels[p].text, to: adds[p].text })
    dels.slice(pairs).forEach((d) => out.push(d))
    adds.slice(pairs).forEach((d) => out.push(d))
  }
  return out
}

/** Common prefix/suffix split for highlighting the changed part of a line. */
export function inlineChange(from: string, to: string): { pre: string; a: string; b: string; post: string } {
  let s = 0
  while (s < from.length && s < to.length && from[s] === to[s]) s++
  let e = 0
  while (e < from.length - s && e < to.length - s && from[from.length - 1 - e] === to[to.length - 1 - e]) e++
  return { pre: from.slice(0, s), a: from.slice(s, from.length - e), b: to.slice(s, to.length - e), post: from.slice(from.length - e) }
}

export function diffStats(rows: DiffRow[]): { added: number; removed: number } {
  let added = 0
  let removed = 0
  for (const r of rows) {
    if (r.type === 'add') added++
    else if (r.type === 'del') removed++
    else if (r.type === 'change') {
      added++
      removed++
    }
  }
  return { added, removed }
}

/** A pending proposal that fixes a skill after a failed run (`origin.repair`). */
export function isRepairProposal(skill: Skill): boolean {
  const o = skill.origin as { repair?: boolean } | 'repair' | null | undefined
  return o === 'repair' || (!!o && typeof o === 'object' && o.repair === true)
}
