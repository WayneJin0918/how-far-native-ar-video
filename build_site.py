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
        lambda m: f'<a class="cite" href="#ref-{int(m.group(1))}" data-ext="{m.group(2)}">{m.group(1)}</a>',
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


def fig_see(zh: bool) -> str:
    if zh:
        cap = "写当前帧时，训练条件看见谁。"
        bi = "短片 · 双向"
        ar = "流 · 自回归"
        past, now, future = "已发生", "正在写", "还没有"
        note_bi = "短片训练里，未来帧和当前帧在同一个样本里。写第 5 帧时，1–8 都可以进条件。"
        note_ar = "流要求的条件只看见已经发生的过去。写第 5 帧时，6–8 还不存在。"
        legend = f'<div class="see-legend"><span class="sw past"></span>{past}<span class="sw now"></span>{now}<span class="sw future"></span>{future}</div>'
    else:
        cap = "What the training condition can see when writing the current frame."
        bi = "short clip · bidirectional"
        ar = "stream · autoregressive"
        past, now, future = "past", "writing", "not yet"
        note_bi = "On a short clip, future frames sit in the same example. Writing frame 5, 1–8 are all legal."
        note_ar = "A stream only admits the past that has already occurred. Writing frame 5, 6–8 do not exist."
        legend = f'<div class="see-legend"><span class="sw past"></span>{past}<span class="sw now"></span>{now}<span class="sw future"></span>{future}</div>'

    cells = []
    for i in range(1, 9):
        kind = "past" if i < 5 else "now" if i == 5 else "future"
        cells.append(f'<div class="frame {kind}"><span>{i}</span></div>')
    frames = "\n          ".join(cells)
    return f"""<figure class="fig fig-see" data-mode="bi" data-note-bi="{html.escape(note_bi)}" data-note-ar="{html.escape(note_ar)}">
  <figcaption class="fig-cap">{html.escape(cap)}</figcaption>
  <div class="see-toggle" role="tablist">
    <button type="button" class="on" data-mode="bi">q<sub>bi</sub> · {html.escape(bi)}</button>
    <button type="button" data-mode="ar">q<sub>AR</sub> · {html.escape(ar)}</button>
  </div>
  <div class="see-axis">
    <div class="lab">t</div>
    <div class="frames">
          {frames}
    </div>
  </div>
  {legend}
  <p class="see-note">{html.escape(note_bi)}</p>
</figure>"""


def fig_loops(zh: bool) -> str:
    if zh:
        cap = "一次前向写什么。左边仍是一段积分；右边才是定义要的因子。"
        left_h, right_h = "现在通行", "原生"
        left = """for i in 1..N:
    x ~ N(0, I)
    for σ in [1.0, 0.75, 0.50, 0.25]:
        x = x + Δσ · v_θ(x, σ | B_<i, x)
    cache.write(x)"""
        right = """for t in 1..T:
    v_t = f_θ(v_<t, c_≤t)   # one forward
    cache.write(v_t)"""
    else:
        cap = "What one forward emits. The left side is still an integral; the right side is the factor the definition asks for."
        left_h, right_h = "Current stack", "Native"
        left = """for i in 1..N:
    x ~ N(0, I)
    for σ in [1.0, 0.75, 0.50, 0.25]:
        x = x + Δσ · v_θ(x, σ | B_<i, x)
    cache.write(x)"""
        right = """for t in 1..T:
    v_t = f_θ(v_<t, c_≤t)   # one forward
    cache.write(v_t)"""
    return f"""<figure class="fig fig-code">
  <figcaption class="fig-cap">{html.escape(cap)}</figcaption>
  <div class="code-cols">
    <div class="code-col">
      <header>{html.escape(left_h)}</header>
      <pre><code>{html.escape(left)}</code></pre>
    </div>
    <div class="code-col native">
      <header>{html.escape(right_h)}</header>
      <pre><code>{html.escape(right)}</code></pre>
    </div>
  </div>
</figure>"""


def fig_stack(zh: bool) -> str:
    if zh:
        cap = "缺口露出来一层，就补一层。点开看每一级在补什么。"
        foot = "如果预训练从来没用过未来，并且一次前向就写出一个因子，这些级没有要补的东西。"
        steps = [
            ("预训练", "双向 DiT，整段短片", "学的是 q_bi，不是 p(v_t | v_<t)。"),
            ("强迫", "块因果 mask，干净历史", "推理时删掉训练用过的未来；历史仍是数据。"),
            ("短程", "自己滚出来的历史 + DMD", "补 K，也补 teacher forcing；历史仍短。"),
            ("长程", "拼接 + 固定窗口", "补长度。训练条件还是没变。"),
            ("搜索", "窗口多长、要不要重置位置", "改的是 H_i，不是模型参数。"),
        ]
    else:
        cap = "The stack is patched in the order the gaps show up. Click a stage."
        foot = "If pretraining never used the future, and one forward emits one factor, these stages have nothing to patch."
        steps = [
            ("pretrain", "bidirectional DiT, full short clip", "Fits q_bi, not p(v_t | v_<t)."),
            ("force", "block-causal mask, clean history", "Deletes at test time variables training used. History is still data."),
            ("short", "self-rollout + DMD", "Patches K, and the teacher-forcing gap. History is still short."),
            ("long", "stitch + fixed window", "Patches length. The training condition is unchanged."),
            ("search", "window size, whether to reset position", "Edits H_i. Still not parameters."),
        ]
    items = []
    for i, (k, v, why) in enumerate(steps):
        on = ' class="on"' if i == 0 else ""
        items.append(
            f"<li{on}>\n"
            f'      <span class="k">{html.escape(k)}</span>\n'
            f'      <span class="v">{html.escape(v)}</span>\n'
            f'      <p class="why">{html.escape(why)}</p>\n'
            f"    </li>"
        )
    return f"""<figure class="fig fig-stack">
  <figcaption class="fig-cap">{html.escape(cap)}</figcaption>
  <ol class="stack">
    {chr(10).join(items)}
  </ol>
  <p class="stack-foot">{html.escape(foot)}</p>
</figure>"""


def fig_exp(zh: bool) -> str:
    if zh:
        cap = "两组未公开实验只保留判断：改块内目标，或只改推理时看得见的过去。"
        rows = (
            "<tr><th></th><th>块内观感</th><th>跨块一致性</th></tr>"
            "<tr><td>更晚的后训练</td><td class=\"up\">同一块里更干净</td>"
            "<td class=\"flat\">没有稳定跟着变好</td></tr>"
            "<tr><td>只改推理窗口 / 重置</td><td>—</td>"
            "<td class=\"flat\">没有稳定跟着变好</td></tr>"
        )
    else:
        cap = "Judgments only from the two unpublished probes: change the intra-block objective, or change only who is visible at test time."
        rows = (
            "<tr><th></th><th>Appearance inside a block</th><th>Consistency across blocks</th></tr>"
            "<tr><td>Later post-training</td><td class=\"up\">Cleaner in the same block</td>"
            "<td class=\"flat\">No reliable gain</td></tr>"
            "<tr><td>Inference window / reset only</td><td>—</td>"
            "<td class=\"flat\">No reliable gain</td></tr>"
        )
    return f"""<figure class="fig fig-exp">
  <figcaption class="fig-cap">{html.escape(cap)}</figcaption>
  <table class="matrix">{rows}</table>
</figure>"""


def classify_code(body: str) -> str | None:
    head = body.lstrip()
    if "diffusion AR" in head or "扩散自回归" in head:
        return "loops"
    if head.startswith("pretrain") or head.startswith("预训练"):
        return "stack"
    return None


def md_to_html(md: str, zh: bool) -> str:
    lines = md.splitlines()
    chunks: list[str] = []
    i = 0
    in_code = False
    code: list[str] = []
    refs = False
    ref_items: list[str] = []
    inserted_see = False
    inserted_exp = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                body = "\n".join(code)
                kind = classify_code(body)
                if kind == "loops":
                    chunks.append(fig_loops(zh))
                elif kind == "stack":
                    chunks.append(fig_stack(zh))
                else:
                    chunks.append("<pre><code>" + html.escape(body) + "</code></pre>")
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
        if line.strip() in {"## References", "## 参考文献"}:
            refs = True
            title = "参考文献" if zh else "References"
            chunks.append(f'<h2 id="references">{title}</h2>')
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
            if not inserted_see and r"q_{\mathrm{AR}}" in "\n".join(block):
                chunks.append(fig_see(zh))
                inserted_see = True
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
        text = " ".join(p.strip() for p in para)
        chunks.append("<p>" + format_inline(text) + "</p>")
        if not inserted_exp and (
            text.startswith("If post-training optimizes")
            or text.startswith("后训练如果优化")
        ):
            chunks.append(fig_exp(zh))
            inserted_exp = True

    if ref_items:
        chunks.append('<div class="refs"><ol>' + "\n".join(ref_items) + "</ol></div>")
    return "\n".join(chunks)


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
  <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600&family=Source+Sans+3:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <link rel="stylesheet" href="css/blog.css">
</head>
<body>
  <div class="wrap">
  <nav class="bar">
    <div class="lang">
      <a href="index.html" class="{en_on}">English</a>
      <a href="zh.html" class="{zh_on}">中文</a>
    </div>
  </nav>
  <article>
    <p class="date">{html.escape(cfg["date"])}</p>
    <h1>{html.escape(cfg["title"])}</h1>
    {body}
  </article>
  </div>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script defer src="js/math.js"></script>
  <script defer src="js/ui.js"></script>
</body>
</html>
"""


def main() -> None:
    (ROOT / ".nojekyll").write_text("")
    for key, cfg in PAGES.items():
        body = md_to_html(cfg["src"].read_text(encoding="utf-8"), zh=key == "zh")
        cfg["out"].write_text(page_html(cfg, body), encoding="utf-8")
        print("wrote", cfg["out"])


if __name__ == "__main__":
    main()
