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


TOC_LABEL = {
    "an-image-has-no-time-video-adds-t": "Time in video",
    "training-sees-the-future-and-the-next-step-is-not-one-forward": "Training and one forward",
    "in-our-experience": "What we saw",
    "what-that-means-for-a-native-design": "A native design",
    "conclusion": "Conclusion",
    "references": "References",
    "cite": "Cite",
    "图像没有时间-视频多出来的是-t": "图像与时间",
    "训练看见了未来-生成一步也不是一次网络计算": "训练与前向",
    "过去的经验": "过去的经验",
    "对原生设计意味着什么": "原生设计",
    "结论": "结论",
}


def toc_html(body: str, zh: bool) -> str:
    zh_fixed = {"references": "参考文献", "cite": "引用"}
    items = [("top", "引言" if zh else "Introduction")]
    for sid, raw in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.S):
        label = (zh_fixed.get(sid) if zh else None) or TOC_LABEL.get(sid) or re.sub(r"<[^>]+>", "", raw)
        items.append((sid, label))
    if not any(sid == "cite" for sid, _ in items):
        items.append(("cite", "引用" if zh else "Cite"))
    lis = "\n".join(
        f'<li><a href="#{html.escape(sid)}">{html.escape(label)}</a></li>' for sid, label in items
    )
    head = "目录" if zh else "Contents"
    return f'<nav class="toc" aria-label="{head}"><p class="toc-k">{head}</p><ol>{lis}</ol></nav>'


def fig_see(zh: bool) -> str:
    if zh:
        note_bi = "短片训练里写第 5 帧，1–8 都可以进条件。"
        note_ar = "流只看见已经发生的。写第 5 帧时，6–8 还不存在。"
    else:
        note_bi = "On a short clip, writing frame 5, 1–8 are all legal."
        note_ar = "A stream only admits the past. Writing frame 5, 6–8 do not exist."

    cells = []
    for i in range(1, 9):
        kind = "past" if i < 5 else "now" if i == 5 else "future"
        cells.append(f'<div class="frame {kind}"><span>{i}</span></div>')
    frames = "\n          ".join(cells)
    return f"""<figure class="fig fig-see" data-mode="bi" data-note-bi="{html.escape(note_bi)}" data-note-ar="{html.escape(note_ar)}">
  <div class="see-toggle" role="tablist">
    <button type="button" class="on" data-mode="bi">q<sub>bi</sub></button>
    <button type="button" data-mode="ar">q<sub>AR</sub></button>
  </div>
  <div class="see-axis">
    <div class="lab">t</div>
    <div class="frames">
          {frames}
    </div>
  </div>
  <p class="see-note">{html.escape(note_bi)}</p>
</figure>"""


def fig_loops(zh: bool) -> str:
    if zh:
        left_h, right_h = "现在通行", "原生"
        left = """for i in 1..N:
    x ~ N(0, I)
    for σ in [1.0, 0.75, 0.50, 0.25]:
        x = x + Δσ · v_θ(x, σ | B_<i, x)
    cache.write(x)"""
        right = """for t in 1..T:
    v_t = f_θ(v_<t, c_≤t)
    cache.write(v_t)"""
    else:
        left_h, right_h = "Current stack", "Native"
        left = """for i in 1..N:
    x ~ N(0, I)
    for σ in [1.0, 0.75, 0.50, 0.25]:
        x = x + Δσ · v_θ(x, σ | B_<i, x)
    cache.write(x)"""
        right = """for t in 1..T:
    v_t = f_θ(v_<t, c_≤t)
    cache.write(v_t)"""
    return f"""<figure class="fig fig-code">
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
        steps = [
            ("预训练", "双向 DiT，整段短片", "学的是 q_bi，不是 p(v_t | v_<t)。"),
            ("强迫", "块因果 mask，干净历史", "推理时删掉训练用过的未来；历史仍是数据。"),
            ("短程", "自己滚出来的历史 + DMD", "补 K，也补 teacher forcing；历史仍短。"),
            ("长程", "拼接 + 固定窗口", "补长度。训练条件还是没变。"),
            ("搜索", "窗口多长、要不要重置位置", "改的是 H_i，不是模型参数。"),
        ]
    else:
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
  <ol class="stack">
    {chr(10).join(items)}
  </ol>
</figure>"""


def fig_exp(zh: bool) -> str:
    if zh:
        rows = (
            "<tr><th></th><th>块内观感</th><th>跨块一致性</th></tr>"
            "<tr><td>更晚的后训练</td><td class=\"up\">同一块里更干净</td>"
            "<td class=\"flat\">没有稳定跟着变好</td></tr>"
            "<tr><td>只改推理窗口 / 重置</td><td>—</td>"
            "<td class=\"flat\">没有稳定跟着变好</td></tr>"
        )
    else:
        rows = (
            "<tr><th></th><th>Appearance inside a block</th><th>Consistency across blocks</th></tr>"
            "<tr><td>Later post-training</td><td class=\"up\">Cleaner in the same block</td>"
            "<td class=\"flat\">No reliable gain</td></tr>"
            "<tr><td>Inference window / reset only</td><td>—</td>"
            "<td class=\"flat\">No reliable gain</td></tr>"
        )
    return f"""<figure class="fig fig-exp">
  <div class="matrix-wrap"><table class="matrix">{rows}</table></div>
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
    bibtex = """@misc{jin2026nativear,
  author       = {Jin, Weiyang},
  title        = {How Far We Are From a Native Autoregressive Video Model},
  year         = {2026},
  month        = aug,
  institution  = {The University of Hong Kong},
  howpublished = {Blog post},
  url          = {https://waynejin0918.github.io/how-far-native-ar-video/},
}"""
    if cfg["lang"].startswith("zh"):
        byline = "Weiyang Jin，香港大学"
        cite_h = "引用"
        copy_label = "复制"
        credit = (
            '版式参考了 '
            '<a href="https://thinkingmachines.ai/blog/on-policy-distillation/">Thinking Machines Lab</a> '
            "的研究笔记，谨此致谢。"
        )
    else:
        byline = "Weiyang Jin, The University of Hong Kong"
        cite_h = "Cite"
        copy_label = "Copy"
        credit = (
            'Visual style after '
            '<a href="https://thinkingmachines.ai/blog/on-policy-distillation/">Thinking Machines Lab</a>.'
        )
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
  <link rel="stylesheet" href="css/blog.css?v=11">
</head>
<body>
  <div class="shell">
    {toc_html(body, cfg["lang"].startswith("zh"))}
    <main class="page">
    <header class="mast">
      <p class="date">{html.escape(cfg["date"])}</p>
      <nav class="lang">
        <a href="index.html" class="{en_on}">English</a>
        <a href="zh.html" class="{zh_on}">中文</a>
      </nav>
    </header>
    <article>
      <h1 id="top">{html.escape(cfg["title"])}</h1>
      <p class="byline">{html.escape(byline)}</p>
      {body}
      <h2 id="cite">{cite_h}</h2>
      <div class="bib">
        <button type="button" class="bib-copy" data-copy>{copy_label}</button>
        <pre><code>{html.escape(bibtex)}</code></pre>
      </div>
    </article>
    <footer class="credit">{credit}</footer>
    </main>
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
