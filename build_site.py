#!/usr/bin/env python3
"""Build bilingual GitHub Pages HTML from the essay markdown."""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

PAGES = {
    "en": {
        "src": SRC / "how_far_native_ar_video.md",
        "out": ROOT / "index.html",
        "title": "How Far We Are From a Native Autoregressive Video Model",
        "lang": "en",
        "date": "31 August 2026",
    },
    "zh": {
        "src": SRC / "how_far_native_ar_video.zh.md",
        "out": ROOT / "zh.html",
        "title": "我们离原生自回归视频模型还有多远",
        "lang": "zh-Hans",
        "date": "2026年8月31日",
    },
}


def cite(n: str) -> str:
    return f'<a class="cite" href="#ref-{int(n)}">{n}</a>'


def format_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(
        r"\[\[(\d+)\]\]\((https?://[^)]+)\)",
        lambda m: f'<a class="cite" href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r"\[(\d+)\]–\[(\d+)\]", lambda m: cite(m.group(1)) + "–" + cite(m.group(2)), text)
    text = re.sub(r"\[(\d+)\]", lambda m: cite(m.group(1)), text)
    text = text.replace("</a><a class=\"cite\"", "</a> <a class=\"cite\"")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def slugify(title: str) -> str:
    title = re.sub(r"\\[()]", "", title)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "-", title.lower()).strip("-") or "section"


def wrap_code_pairs(chunks: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(chunks):
        if (
            chunks[i].startswith("<pre>")
            and i + 1 < len(chunks)
            and chunks[i + 1].startswith("<pre>")
        ):
            out.append('<div class="code-pair">\n' + chunks[i] + "\n" + chunks[i + 1] + "\n</div>")
            i += 2
            continue
        out.append(chunks[i])
        i += 1
    return out


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    chunks: list[str] = []
    i = 0
    in_code = False
    code: list[str] = []
    refs = False
    ref_items: list[str] = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                chunks.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if line.strip() in {"---", ""}:
            i += 1
            continue
        if line.startswith("# ") and not line.startswith("## "):
            i += 1
            continue
        if line.strip() == "## References":
            refs = True
            chunks.append('<h2 id="references">References</h2>')
            i += 1
            continue
        if refs:
            m = re.match(r"^(\d+)\.\s+(.*)$", line)
            if m:
                n, rest = m.group(1), m.group(2)
                rest = html.escape(rest)
                rest = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', rest)
                rest = re.sub(r"\*(.+?)\*", r"<em>\1</em>", rest)
                ref_items.append(f'<li id="ref-{n}">{rest}</li>')
            elif line.startswith("    ") and ref_items:
                extra = html.escape(line.strip())
                extra = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', extra)
                extra = re.sub(r"\*(.+?)\*", r"<em>\1</em>", extra)
                ref_items[-1] = ref_items[-1][:-5] + "<br>" + extra + "</li>"
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            chunks.append(f'<h2 id="{slugify(title)}">{format_inline(title)}</h2>')
            i += 1
            continue
        if line.startswith("\\[") or line.strip() == "\\[":
            block = [line]
            if line.strip() != "\\]" and not line.strip().endswith("\\]"):
                i += 1
                while i < len(lines) and lines[i].strip() != "\\]":
                    block.append(lines[i])
                    i += 1
                if i < len(lines):
                    block.append(lines[i])
            chunks.append('<div class="math-block">' + html.escape("\n".join(block)) + "</div>")
            i += 1
            continue
        para = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith("#")
            and not lines[i].startswith("```")
            and lines[i].strip() != "---"
            and not lines[i].startswith("\\[")
        ):
            para.append(lines[i])
            i += 1
        chunks.append("<p>" + format_inline(" ".join(p.strip() for p in para)) + "</p>")

    if ref_items:
        chunks.append('<div class="refs"><ol>' + "\n".join(ref_items) + "</ol></div>")
    return "\n".join(wrap_code_pairs(chunks))


def page_html(cfg: dict, body: str) -> str:
    en_on = "on" if cfg["lang"] == "en" else ""
    zh_on = "on" if cfg["lang"].startswith("zh") else ""
    return f"""<!DOCTYPE html>
<html lang="{cfg["lang"]}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(cfg["title"])}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://cdn.jsdelivr.net">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <link rel="stylesheet" href="css/blog.css">
</head>
<body>
  <nav class="bar">
    <a href="index.html" class="{en_on}">English</a>
    <a href="zh.html" class="{zh_on}">中文</a>
  </nav>
  <article>
    <p class="date">{html.escape(cfg["date"])}</p>
    <h1>{html.escape(cfg["title"])}</h1>
    {body}
  </article>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script defer src="js/math.js"></script>
</body>
</html>
"""


def main() -> None:
    (ROOT / ".nojekyll").write_text("")
    for cfg in PAGES.values():
        body = md_to_html(cfg["src"].read_text(encoding="utf-8"))
        cfg["out"].write_text(page_html(cfg, body), encoding="utf-8")
        print("wrote", cfg["out"])


if __name__ == "__main__":
    main()
