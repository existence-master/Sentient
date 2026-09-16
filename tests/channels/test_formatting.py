from __future__ import annotations

import random

from sentient.channels.formatting import (
    TELEGRAM_LIMIT,
    html_to_plain,
    markdown_to_telegram_html,
    render_telegram_chunks,
    split_markdown,
    strip_links,
    summary_text,
    tags_balanced,
)


def test_escapes_everything_else():
    assert markdown_to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"
    out = markdown_to_telegram_html("<script>alert(1)</script> <b>not bold</b>")
    assert "<script>" not in out and "<b>" not in out and "&lt;b&gt;" in out


def test_inline_constructs():
    out = markdown_to_telegram_html(
        "**bold** and *it* and _also_ and ~~gone~~ and `x<y>` see [site](https://e.com/?a=1&b=2)"
    )
    assert "<b>bold</b>" in out
    assert "<i>it</i>" in out and "<i>also</i>" in out
    assert "<s>gone</s>" in out
    assert "<code>x&lt;y&gt;</code>" in out
    assert '<a href="https://e.com/?a=1&amp;b=2">site</a>' in out


def test_snake_case_and_math_are_not_formatted():
    assert markdown_to_telegram_html("use file_name_here and 2*3*4") == "use file_name_here and 2*3*4"


def test_unsafe_links_are_not_linked():
    out = markdown_to_telegram_html("[click](javascript:alert(1))")
    assert "<a" not in out and "click" in out


def test_code_blocks_are_escaped_and_not_formatted():
    out = markdown_to_telegram_html("Here:\n```python\nif a<b:\n    print('**no**')\n```\ndone")
    assert '<pre><code class="language-python">if a&lt;b:\n    print(\'**no**\')</code></pre>' in out
    assert "<b>" not in out and out.endswith("done")


def test_unclosed_fence_while_streaming_is_balanced():
    out = markdown_to_telegram_html("```\npartial code **")
    assert out.startswith("<pre><code>") and tags_balanced(out)


def test_blocks_lists_headings_quotes_tables():
    out = markdown_to_telegram_html(
        "# Title\n- one\n  - nested **b**\n* [x] done\n1. three\n> quoted\n> more\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    )
    assert out.startswith("<b>Title</b>")
    assert "• one" in out and "  • nested <b>b</b>" in out and "☑ done" in out and "1. three" in out
    assert "<blockquote>quoted\nmore</blockquote>" in out
    assert "<pre>| a | b |\n| 1 | 2 |</pre>" in out


def test_random_markdown_is_always_balanced():
    rng = random.Random(7)
    alphabet = ["*", "**", "_", "__", "`", "```", "[", "]", "(", ")", "https://x.y", "~~", "<", ">", "&", "a", " ", "\n", "#", "- ", "> "]
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        out = markdown_to_telegram_html(text)
        assert tags_balanced(out), (text, out)
        assert "<script" not in out


def test_html_to_plain():
    assert html_to_plain('<b>Hi</b> &amp; <a href="https://e.com">site</a>') == "Hi & site (https://e.com)"


def test_split_prefers_paragraphs_and_keeps_limit():
    paras = [("word " * 150).strip() for _ in range(20)]
    text = "\n\n".join(paras)
    pieces = split_markdown(text, 3000)
    assert len(pieces) > 1 and all(len(p) <= 3000 for p in pieces)
    assert all(p.endswith("word") for p in pieces)


def test_split_reopens_code_fences():
    code = "\n".join(f"line {i} = value" for i in range(700))
    pieces = split_markdown(f"Intro\n```python\n{code}\n```\nOutro", 2000)
    assert len(pieces) > 2
    for p in pieces:
        assert len(p) <= 2000
        assert p.count("```") % 2 == 0
    assert pieces[1].startswith("```python")


def test_render_chunks_respects_limit_after_escaping():
    chunks = render_telegram_chunks("& < > " * 3000)
    assert len(chunks) > 1 and all(len(c) <= TELEGRAM_LIMIT for c in chunks)
    chunks = render_telegram_chunks("\n\n".join(["**bold** text " * 40] * 30))
    assert all(len(c) <= TELEGRAM_LIMIT and tags_balanced(c) for c in chunks)


def test_summary_is_link_free_and_short():
    assert strip_links("See [the doc](https://x.com/a) at https://y.com/b now") == "See the doc at now"
    long = summary_text("word " * 500, 100)
    assert len(long) <= 102 and long.endswith("…")
