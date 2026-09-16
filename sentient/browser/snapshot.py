"""Page snapshot and readable-text extraction.

The scripts run inside the page. ``SNAPSHOT_JS`` tags every visible interactive element with a
``data-sentient-ref`` attribute (``e1``, ``e2`` ...) and returns a descriptor for each, plus a
readable text summary. Python turns that into the compact listing the model reads::

    [e12] button "Sign in"
    [e4] textbox "Search" value=""

Formatting lives here as pure functions so it can be tested without a browser.
"""

from __future__ import annotations

import re

REF_ATTR = "data-sentient-ref"

# Shared by both scripts: readable text of a root, skipping navigation chrome and hidden nodes.
_READABLE_JS = r"""
const SKIP = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','NAV','HEADER','FOOTER','ASIDE','IFRAME','TEMPLATE',
  'CANVAS','OBJECT','EMBED','SELECT','OPTION','BUTTON','DIALOG']);
const BLOCK = new Set(['P','DIV','BR','LI','UL','OL','TR','TABLE','SECTION','ARTICLE','MAIN','H1','H2','H3','H4',
  'H5','H6','BLOCKQUOTE','PRE','HR','DD','DT','FIGCAPTION','FORM','FIELDSET','LABEL','TD','TH']);
function hiddenEl(el) {
  if (el.getAttribute && (el.getAttribute('aria-hidden') === 'true' || el.hidden)) return true;
  const role = el.getAttribute && el.getAttribute('role');
  if (role === 'navigation' || role === 'banner' || role === 'contentinfo') return true;
  const cs = window.getComputedStyle(el);
  return cs.display === 'none' || cs.visibility === 'hidden';
}
function readable(root) {
  const out = [];
  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) { out.push(node.textContent); return; }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const tag = node.tagName.toUpperCase();
    if (SKIP.has(tag) || hiddenEl(node)) return;
    const block = BLOCK.has(tag);
    if (block) out.push('\n');
    if (tag === 'H1' || tag === 'H2' || tag === 'H3') out.push('#'.repeat(Number(tag[1])) + ' ');
    if (tag === 'LI') out.push('- ');
    if (tag === 'IMG' && node.alt) out.push(' ' + node.alt + ' ');
    for (const child of node.childNodes) walk(child);
    if (block) out.push('\n');
  }
  walk(root);
  return out.join('').split('\n').map(l => l.replace(/[ \t\r\f\v\u00a0]+/g, ' ').trim())
    .filter((l, i, arr) => l || (i > 0 && arr[i - 1])).join('\n').replace(/\n{3,}/g, '\n\n').trim();
}
function mainRoot() {
  const cands = Array.from(document.querySelectorAll('main, [role=main], article'));
  let best = null, bestLen = 0;
  for (const c of cands) { const n = (c.innerText || '').length; if (n > bestLen) { best = c; bestLen = n; } }
  const bodyLen = (document.body && document.body.innerText || '').length;
  return best && bestLen > Math.min(400, bodyLen * 0.3) ? best : document.body;
}
"""

SNAPSHOT_JS = (
    "(args) => {\n"
    + _READABLE_JS
    + r"""
const REF = 'data-sentient-ref';
document.querySelectorAll('[' + REF + ']').forEach(el => el.removeAttribute(REF));
const SELECTOR = [
  'a[href]', 'button', 'input:not([type=hidden])', 'select', 'textarea', 'summary', '[contenteditable=""]',
  '[contenteditable=true]', '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]', '[role=tab]',
  '[role=menuitem]', '[role=menuitemcheckbox]', '[role=menuitemradio]', '[role=option]', '[role=switch]',
  '[role=combobox]', '[role=textbox]', '[role=searchbox]', '[role=slider]', '[role=spinbutton]', '[role=treeitem]'
].join(',');
const clip = (s, n) => { s = (s || '').replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
function visible(el) {
  const r = el.getBoundingClientRect();
  if (r.width < 1 || r.height < 1) return false;
  const cs = window.getComputedStyle(el);
  if (cs.visibility === 'hidden' || cs.display === 'none' || Number(cs.opacity) === 0) return false;
  return !el.closest('[aria-hidden=true], [hidden], [inert]');
}
function labelFor(el) {
  let text = '';
  if (el.labels && el.labels.length) text = Array.from(el.labels).map(l => l.innerText).join(' ');
  const by = el.getAttribute('aria-labelledby');
  if (!text && by) text = by.split(/\s+/).map(id => { const n = document.getElementById(id); return n ? n.innerText : ''; }).join(' ');
  return clip(text, 80);
}
function roleOf(el) {
  const explicit = el.getAttribute('role');
  if (explicit) return explicit.split(/\s+/)[0];
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  if (tag === 'a') return 'link';
  if (tag === 'button' || tag === 'summary') return 'button';
  if (tag === 'select') return el.multiple ? 'listbox' : 'combobox';
  if (tag === 'textarea' || el.isContentEditable) return 'textbox';
  if (tag === 'input') {
    if (['button', 'submit', 'reset', 'image'].includes(type)) return 'button';
    if (type === 'checkbox') return 'checkbox';
    if (type === 'radio') return 'radio';
    if (type === 'range') return 'slider';
    if (type === 'search') return 'searchbox';
    if (type === 'file') return 'file';
    if (['date', 'datetime-local', 'month', 'week', 'time', 'color'].includes(type)) return type;
    return 'textbox';
  }
  return tag;
}
function nameOf(el, label) {
  const aria = el.getAttribute('aria-label');
  if (aria) return clip(aria, 80);
  if (label) return label;
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  if (tag === 'input' && ['button', 'submit', 'reset'].includes(type)) return clip(el.value || type, 80);
  if (tag === 'input' || tag === 'textarea' || tag === 'select') {
    return clip(el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || '', 80);
  }
  let text = clip(el.innerText, 80);
  if (!text) { const img = el.querySelector('img[alt]'); if (img) text = clip(img.alt, 80); }
  return text || clip(el.getAttribute('title') || '', 80);
}
function formInfo(el) {
  const form = el.form || el.closest('form');
  if (!form) return { in_form: false, form_payment: false, form_submit_text: '' };
  const fields = Array.from(form.querySelectorAll('input, select, textarea'));
  const pay = /\b(cc|card|cvv|cvc|csc|credit|debit|expir|upi|iban|payment)/i;
  const payment = fields.some(f => {
    const ac = (f.getAttribute('autocomplete') || '').toLowerCase();
    if (/(^|\s)cc-/.test(ac)) return true;
    return pay.test([f.name, f.id, f.getAttribute('placeholder'), f.getAttribute('aria-label')].join(' '));
  });
  const submits = Array.from(form.querySelectorAll('button:not([type=button]):not([type=reset]), input[type=submit], input[type=image]'));
  const submitText = clip(submits.map(b => b.innerText || b.value || b.getAttribute('aria-label') || '').join(' | '), 120);
  return { in_form: true, form_payment: payment, form_submit_text: submitText };
}
const els = [];
let n = 0;
const max = args.maxElements || 300;
for (const el of document.querySelectorAll(SELECTOR)) {
  if (els.length >= max) break;
  if (!visible(el)) continue;
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  const role = roleOf(el);
  const label = labelFor(el);
  const ref = 'e' + (++n);
  el.setAttribute(REF, ref);
  const isField = tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
  const d = {
    ref, role, tag, type, name: nameOf(el, label), label,
    aria_label: clip(el.getAttribute('aria-label'), 80), title: clip(el.getAttribute('title'), 80),
    placeholder: clip(el.getAttribute('placeholder'), 80), autocomplete: el.getAttribute('autocomplete') || '',
    id: el.id || '', name_attr: el.getAttribute('name') || '',
    disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
    is_submit: (tag === 'button' && (type === '' || type === 'submit') && !!(el.form)) || (tag === 'input' && (type === 'submit' || type === 'image')),
  };
  if (tag === 'a') { d.href = clip(el.getAttribute('href'), 100); if (el.getAttribute('target')) d.target = el.getAttribute('target'); }
  if (isField && type !== 'checkbox' && type !== 'radio' && !['button', 'submit', 'reset', 'image'].includes(type)) {
    d.value = type === 'password' ? '' : clip(el.isContentEditable ? el.innerText : el.value, 60);
  }
  if (type === 'checkbox' || type === 'radio') d.checked = !!el.checked;
  else if (el.getAttribute('aria-checked')) d.checked = el.getAttribute('aria-checked') === 'true';
  if (el.getAttribute('aria-expanded')) d.expanded = el.getAttribute('aria-expanded') === 'true';
  if (tag === 'select') {
    d.options = Array.from(el.options).slice(0, 12).map(o => clip(o.label || o.text, 40));
    d.more_options = Math.max(0, el.options.length - 12);
    const sel = el.selectedOptions && el.selectedOptions[0];
    d.value = sel ? clip(sel.label || sel.text, 60) : '';
  }
  Object.assign(d, formInfo(el));
  d.focused = document.activeElement === el;
  els.push(d);
}
const root = mainRoot();
return {
  url: location.href, title: document.title, elements: els,
  text: readable(root || document.body).slice(0, args.maxText || 20000),
  scroll: { y: Math.round(window.scrollY), height: Math.round(document.documentElement.scrollHeight), view: window.innerHeight },
};
}"""
)

EXTRACT_JS = (
    "(args) => {\n"
    + _READABLE_JS
    + r"""
const root = mainRoot();
return { url: location.href, title: document.title, text: readable(root || document.body).slice(0, args.maxText || 500000) };
}"""
)

# Live descriptor for one element (used right before acting on it, fresher than the snapshot).
ELEMENT_INFO_JS = r"""(el) => {
  const clip = (s, n) => { s = (s || '').replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n) : s; };
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  let label = '';
  if (el.labels && el.labels.length) label = Array.from(el.labels).map(l => l.innerText).join(' ');
  const form = el.form || el.closest('form');
  let payment = false, submitText = '';
  if (form) {
    const pay = /\b(cc|card|cvv|cvc|csc|credit|debit|expir|upi|iban|payment)/i;
    payment = Array.from(form.querySelectorAll('input, select, textarea')).some(f => {
      const ac = (f.getAttribute('autocomplete') || '').toLowerCase();
      return /(^|\s)cc-/.test(ac) || pay.test([f.name, f.id, f.getAttribute('placeholder'), f.getAttribute('aria-label')].join(' '));
    });
    submitText = clip(Array.from(form.querySelectorAll('button:not([type=button]):not([type=reset]), input[type=submit]'))
      .map(b => b.innerText || b.value || '').join(' | '), 120);
  }
  let name = el.getAttribute('aria-label') || '';
  if (!name && (tag === 'input' || tag === 'textarea' || tag === 'select')) name = label || el.getAttribute('placeholder') || '';
  if (!name) name = el.innerText || el.value || el.getAttribute('title') || '';
  return {
    role: el.getAttribute('role') || tag, tag, type, name: clip(name, 80), label: clip(label, 80),
    aria_label: clip(el.getAttribute('aria-label'), 80), title: clip(el.getAttribute('title'), 80),
    placeholder: clip(el.getAttribute('placeholder'), 80), autocomplete: el.getAttribute('autocomplete') || '',
    id: el.id || '', name_attr: el.getAttribute('name') || '',
    value: ['submit', 'button', 'image', 'reset'].includes(type) ? clip(el.value, 60) : '',
    in_form: !!form, form_payment: payment, form_submit_text: submitText,
    is_submit: (tag === 'button' && (type === '' || type === 'submit') && !!el.form) || (tag === 'input' && (type === 'submit' || type === 'image')),
    disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
    target: el.getAttribute('target') || '',
  };
}"""

FOCUSED_INFO_JS = "() => { const el = document.activeElement; return el && el !== document.body ? (" + ELEMENT_INFO_JS + ")(el) : null; }"


# ----------------------------------------------------------------------------- formatting (pure)
def _q(text: object) -> str:
    return '"' + str(text or "").replace('"', "'") + '"'


def format_element(el: dict) -> str:
    """One compact line for the model, e.g. ``[e4] textbox "Search" value=""``."""
    role = el.get("role") or el.get("tag") or "element"
    line = f"[{el.get('ref', '?')}] {role} {_q(el.get('name'))}"
    ftype = str(el.get("type") or "").lower()
    if ftype == "password":
        line += " (password)"
    if "value" in el and role not in {"link", "button"}:
        line += f" value={_q(el.get('value'))}"
    if "checked" in el:
        line += " checked" if el.get("checked") else " unchecked"
    if "expanded" in el:
        line += " expanded" if el.get("expanded") else " collapsed"
    if el.get("options"):
        opts = " | ".join(str(o) for o in el["options"])
        more = f" (+{el['more_options']} more)" if el.get("more_options") else ""
        line += f" options: {opts}{more}"
    if role == "link" and el.get("href") and not str(el["href"]).startswith("javascript:"):
        line += f" -> {el['href']}"
    if el.get("disabled"):
        line += " disabled"
    if el.get("focused"):
        line += " focused"
    return line


def format_snapshot(title: str, url: str, elements: list[dict], text: str, max_chars: int,
                    scroll: dict | None = None) -> tuple[str, bool]:
    """Header, interactive elements, then page text; capped at ``max_chars``.

    Elements get up to ~65% of the budget so a long article cannot push every button out."""
    header = f"Page: {title or '(untitled)'}\nURL: {url}"
    if scroll and scroll.get("height") and scroll.get("view"):
        below = max(0, int(scroll["height"]) - int(scroll.get("y", 0)) - int(scroll["view"]))
        header += f"\nScroll: {int(scroll.get('y', 0))}px from top, {below}px more below"
    truncated = False
    lines: list[str] = []
    budget_elements = int(max_chars * 0.65)
    used = 0
    for el in elements:
        ln = format_element(el)
        if used + len(ln) + 1 > budget_elements:
            lines.append(f"... {len(elements) - len(lines)} more elements not shown (scroll or use browser_extract)")
            truncated = True
            break
        lines.append(ln)
        used += len(ln) + 1
    body = header + "\n\nInteractive elements:\n" + ("\n".join(lines) if lines else "(none)")
    body += "\n\nPage text:\n"
    remaining = max_chars - len(body)
    clean = collapse_text(text)
    if remaining <= 0:
        return body[:max_chars], True
    if len(clean) > remaining:
        cut = clean[: max(0, remaining - 20)].rstrip()
        clean = cut + "\n...(text cut)"
        truncated = True
    return body + (clean or "(no text)"), truncated


def collapse_text(text: str) -> str:
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in (text or "").split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def focus_on_question(text: str, question: str, max_chars: int) -> tuple[str, bool]:
    """Cap readable text. When it is too long and a question is given, keep the paragraphs that
    share the most words with the question (in page order)."""
    text = collapse_text(text)
    if len(text) <= max_chars:
        return text, False
    words = {w for w in re.findall(r"[a-z0-9]{3,}", (question or "").lower())}
    if not words:
        return text[: max_chars - 15].rstrip() + "\n...(text cut)", True
    paras = [p for p in re.split(r"\n\s*\n|\n(?=#)", text) if p.strip()]
    scored = sorted(
        range(len(paras)),
        key=lambda i: (-len(words & set(re.findall(r"[a-z0-9]{3,}", paras[i].lower()))), i),
    )
    keep: set[int] = set()
    used = 0
    for i in scored:
        if used + len(paras[i]) + 2 > max_chars - 40:
            continue
        keep.add(i)
        used += len(paras[i]) + 2
    picked = "\n\n".join(paras[i] for i in sorted(keep))
    return picked + "\n\n...(only the parts most related to the question are shown)", True
