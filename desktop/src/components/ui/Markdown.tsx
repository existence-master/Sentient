import { IconCheck, IconCopy } from '@tabler/icons-react'
import { memo, useRef, useState, type ComponentProps, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import { getBridge } from '@/lib/bridge'
import { cn, copyText } from '@/lib/utils'

function CodeBlock({ children, ...props }: ComponentProps<'pre'>) {
  const ref = useRef<HTMLPreElement>(null)
  const [copied, setCopied] = useState(false)
  const child = Array.isArray(children) ? children[0] : children
  const className = (child as { props?: { className?: string } } | undefined)?.props?.className ?? ''
  const language = /language-([\w+-]+)/.exec(className)?.[1]

  return (
    <div className="md-code group/code relative overflow-hidden rounded-xl border border-border bg-sunken">
      <div className="flex h-8 items-center justify-between border-b border-border px-3 text-2xs text-fg-subtle">
        <span className="font-mono lowercase">{language ?? 'text'}</span>
        <button
          type="button"
          onClick={async () => {
            if (await copyText(ref.current?.textContent ?? '')) {
              setCopied(true)
              setTimeout(() => setCopied(false), 1400)
            }
          }}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-fg-subtle transition-colors hover:bg-hover hover:text-fg"
        >
          {copied ? <IconCheck size={12} /> : <IconCopy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre ref={ref} {...props} className="!m-0 overflow-x-auto px-3.5 py-3 font-mono text-[12.5px] leading-relaxed">
        {children}
      </pre>
    </div>
  )
}

function ExternalLink({ href, children, ...props }: ComponentProps<'a'>) {
  const safe = href && /^(https?:|mailto:)/i.test(href) ? href : undefined
  return (
    <a
      {...props}
      href={safe}
      title={safe}
      target="_blank"
      rel="noreferrer noopener"
      onClick={(e) => {
        e.preventDefault()
        if (safe) void getBridge().openExternal(safe)
      }}
    >
      {children}
    </a>
  )
}

const components: Components = {
  pre: CodeBlock as Components['pre'],
  a: ExternalLink as Components['a'],
  table: ({ children }) => (
    <div className="overflow-x-auto">
      <table>{children}</table>
    </div>
  ),
  img: ({ src, alt }) => (typeof src === 'string' && /^(https?:|data:image\/)/.test(src) ? <img src={src} alt={alt ?? ''} loading="lazy" /> : null)
}

const remarkPlugins = [remarkGfm]
const rehypePlugins: ComponentProps<typeof ReactMarkdown>['rehypePlugins'] = [[rehypeHighlight, { detect: false }]]

export interface MarkdownProps {
  children: string
  streaming?: boolean
  className?: string
}

/** GFM markdown with highlighted code, copy buttons and links opened in the system browser. */
export const Markdown = memo(function Markdown({ children, streaming, className }: MarkdownProps): ReactNode {
  return (
    <div className={cn('md', streaming && 'md-streaming', className)}>
      <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components} skipHtml>
        {children}
      </ReactMarkdown>
    </div>
  )
})
