// Unit tests for the Skills page's section splitter (#527): headings inside fenced code are part of the example,
// not new section cards. The fence rules match the engine's `_without_fenced_code` in sentient/skills/loader.py.
// Run with `npm test` (Node's test runner with TypeScript type stripping; no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseSections } from '../src/features/skills/sections.ts'

const titles = (body: string) => parseSections(body).sections.map((s) => s.title)

test('a normal skill splits into its sections as before', () => {
  const body = '# Pack for a trip\nHelps Maya pack.\n\n## When to use\n- A trip is coming up\n\n## Procedure\n1. Check the weather\n2. Make the list\n\n### Tip\nRoll the shirts.\n\n## Verification\n- Nothing is missing'
  const parsed = parseSections(body)
  assert.equal(parsed.title, 'Pack for a trip')
  assert.equal(parsed.intro, 'Helps Maya pack.')
  assert.deepEqual(parsed.sections, [
    { title: 'When to use', content: '- A trip is coming up' },
    { title: 'Procedure', content: '1. Check the weather\n2. Make the list\n\n### Tip\nRoll the shirts.' },
    { title: 'Verification', content: '- Nothing is missing' }
  ])
})

test('a heading inside a backtick fence stays in its section', () => {
  const body = '# Packing\n## Procedure\n2. Write the list in this shape:\n```markdown\n## Output\n- shirts x days\n```\n\n## Output\n- The list'
  const parsed = parseSections(body)
  assert.deepEqual(parsed.sections, [
    { title: 'Procedure', content: '2. Write the list in this shape:\n```markdown\n## Output\n- shirts x days\n```' },
    { title: 'Output', content: '- The list' }
  ])
})

test('tilde fences hide headings too', () => {
  assert.deepEqual(titles('## Procedure\n~~~markdown\n## Steps\n# Not a title\n~~~\n## Approval\n- Ask first'), ['Procedure', 'Approval'])
})

test('a fence indented up to three spaces counts, four spaces does not', () => {
  assert.deepEqual(titles('## Procedure\n   ```\n## Steps\n   ```\n## Output'), ['Procedure', 'Output'])
  // four spaces is an indented code line, not a fence, so the heading below it is real
  assert.deepEqual(titles('## Procedure\n    ```\n## Steps\n    ```'), ['Procedure', 'Steps'])
})

test('a fence only closes on the same character at least as long', () => {
  // a shorter run and the other character don't close it, so "## Steps" is still inside
  assert.deepEqual(titles('## Procedure\n````markdown\n```\n~~~~\n## Steps\n````\n## Output'), ['Procedure', 'Output'])
  // a longer run closes it
  assert.deepEqual(titles('## Procedure\n~~~\n## Steps\n~~~~~\n## Output'), ['Procedure', 'Output'])
  // a closing line with text after the marker doesn't close it
  assert.deepEqual(titles('## Procedure\n```\n``` not yet\n## Steps\n```\n## Output'), ['Procedure', 'Output'])
})

test('a fence that is never closed runs to the end', () => {
  const parsed = parseSections('## Procedure\n```markdown\n## Steps\n## Output')
  assert.deepEqual(parsed.sections, [{ title: 'Procedure', content: '```markdown\n## Steps\n## Output' }])
})

test('backticks in the info string mean it is not a fence', () => {
  assert.deepEqual(titles('## Procedure\n``` `code` ```\n## Steps'), ['Procedure', 'Steps'])
})

test('a fence before the first section keeps its heading in the intro', () => {
  const parsed = parseSections('```\n# Example\n## Steps\n```\n# Real title\n## Procedure\n1. Go')
  assert.equal(parsed.title, 'Real title')
  assert.equal(parsed.intro, '```\n# Example\n## Steps\n```')
  assert.deepEqual(parsed.sections, [{ title: 'Procedure', content: '1. Go' }])
})

test('Windows line endings work the same', () => {
  assert.deepEqual(titles('## Procedure\r\n```\r\n## Steps\r\n```\r\n## Output\r\n'), ['Procedure', 'Output'])
})
