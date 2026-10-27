// An MCP app's name lands in the approval and plan-ready notes, which the app renders
// as Markdown (components/ui/Markdown.tsx: react-markdown + remark-gfm + rehype-highlight,
// skipHtml). Backslash escaping cannot stop GFM autolinks: remark-gfm resolves escapes
// before it looks for bare URLs, so `https\://evil.co` still renders as a live link
// (review 5641825878). The engine therefore puts the name in an inline code span, whose
// contents Markdown never parses. These tests render the notes exactly that way and pin
// the result: one <code> per span, never an <a>, <strong> or <img>.
// Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import React, { type ComponentProps } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'

const remarkPlugins = [remarkGfm]
const rehypePlugins: ComponentProps<typeof ReactMarkdown>['rehypePlugins'] = [[rehypeHighlight, { detect: false }]]

function render(note: string): string {
  return renderToStaticMarkup(
    React.createElement(ReactMarkdown, { remarkPlugins, rehypePlugins, skipHtml: true }, note)
  )
}

// The engine's code-span rule (sentient/app_names.py::code_span): a fence one backtick
// longer than the longest run inside, with a space inside each end.
function codeSpan(text: string): string {
  const longest = (text.match(/`+/g) ?? []).reduce((n, run) => Math.max(n, run.length), 0)
  return `${'`'.repeat(longest + 1)} ${text} ${'`'.repeat(longest + 1)}`
}

function approvalNote(appName: string): string {
  return `An app called ${codeSpan(appName)} wants to: Runs code`
}

function count(html: string, tag: string): number {
  return html.split(tag).length - 1
}

// Every hostile name from review 5641825878, plus backticks, entities and other schemes.
const hostileNames = [
  '[Fix it](https://evil.co) **NOW**',
  'https://evil.co/allow',
  'www.evil.co',
  'WWW.EVIL.CO',
  '![x](http://evil.co/p.png)',
  '_https://evil.co',
  'support@evil.co',
  'mailto:evil@evil.co',
  'xmpp:foo@evil.co',
  '<https://evil.co>',
  '&#91;evil&#93;',
  '[^1] see [x][y]',
  'a `b` c',
  '`x'
]

test('an approval note with a hostile app name renders one code span, no link', () => {
  for (const name of hostileNames) {
    const html = render(approvalNote(name))
    assert.equal(count(html, '<code>'), 1, name)
    assert.equal(count(html, '</code>'), 1, name)
    assert.ok(!html.includes('<a '), `${name} -> ${html}`)
    assert.ok(!html.includes('<strong'), `${name} -> ${html}`)
    assert.ok(!html.includes('<img'), `${name} -> ${html}`)
  }
})

test('the maintainer’s example renders its characters, not a link', () => {
  const html = render(approvalNote('[Fix it](https://evil.co) **NOW**'))
  assert.ok(html.includes('<code>[Fix it](https://evil.co) **NOW**</code>'), html)
})

test('a plan-ready note with a hostile name and a hostile title renders two code spans, no link', () => {
  const note = `I've created a new plan for an app called ${codeSpan('www.evil.co')}: ${codeSpan('See [this](https://evil.co) now')}`
  const html = render(note)
  assert.equal(count(html, '<code>'), 2, html)
  assert.ok(!html.includes('<a '), html)
  assert.ok(!html.includes('<strong'), html)
  assert.ok(!html.includes('<img'), html)
})
