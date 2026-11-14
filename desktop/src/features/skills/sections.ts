/** Splitting a SKILL.md body into the section cards the Skills page shows. Pure, so `npm test` can import it. */

export interface SkillSection {
  title: string
  content: string
}

// An opening code fence: up to 3 spaces, then 3+ backticks or 3+ tildes, then an optional info string. The same
// rule as the engine's `_FENCE_OPEN` in sentient/skills/loader.py, so the page and the engine agree on what a
// section heading is.
const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})(.*)$/

interface Fence {
  char: '`' | '~'
  length: number
}

function fenceOpening(line: string): Fence | null {
  const m = FENCE_OPEN.exec(line)
  if (!m) return null
  const char = m[1][0] as Fence['char']
  // CommonMark: a backtick fence's info string can't contain a backtick (that's inline code, not a fence).
  if (char === '`' && m[2].includes('`')) return null
  return { char, length: m[1].length }
}

/** The fence closes on a line of the same character, at least as long, up to 3 spaces in and nothing after it. */
function closesFence(line: string, fence: Fence): boolean {
  const m = /^ {0,3}(`+|~+)[ \t]*$/.exec(line)
  return !!m && m[1][0] === fence.char && m[1].length >= fence.length
}

/**
 * Split a SKILL.md body into its `##` sections (text before the first one is the intro). Headings inside fenced
 * code blocks are part of the example, not new sections, and a fence that never closes runs to the end.
 */
export function parseSections(body: string): { title: string; intro: string; sections: SkillSection[] } {
  const lines = body.split(/\r?\n/)
  let title = ''
  const intro: string[] = []
  const sections: SkillSection[] = []
  let fence: Fence | null = null
  const append = (line: string) => {
    if (sections.length) sections[sections.length - 1].content += `${line}\n`
    else intro.push(line)
  }
  for (const line of lines) {
    if (fence) {
      if (closesFence(line, fence)) fence = null
      append(line)
      continue
    }
    const opening = fenceOpening(line)
    if (opening) {
      fence = opening
      append(line)
      continue
    }
    const h1 = /^#\s+(.+)$/.exec(line)
    const h2 = /^##\s+(.+)$/.exec(line)
    if (h1 && !title && !sections.length) title = h1[1].trim()
    else if (h2) sections.push({ title: h2[1].trim(), content: '' })
    else append(line)
  }
  return { title, intro: intro.join('\n').trim(), sections: sections.map((s) => ({ ...s, content: s.content.trim() })) }
}
