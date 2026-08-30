#!/usr/bin/env python3
"""Build bilingual GitHub Pages HTML from the essay markdown."""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DOCS = ROOT

REF_URL = {
    1: "https://arxiv.org/abs/2608.23189",
    3: "https://arxiv.org/abs/1511.06732",
    4: "https://arxiv.org/abs/1601.06759",
    5: "https://arxiv.org/abs/1610.00527",
    7: "https://arxiv.org/abs/2104.10157",
    8: "https://arxiv.org/abs/2202.04200",
    9: "https://arxiv.org/abs/2204.03638",
    10: "https://arxiv.org/abs/2210.02399",
    11: "https://arxiv.org/abs/2212.05199",
    12: "https://arxiv.org/abs/2212.09748",
    13: "https://arxiv.org/abs/2309.17453",
    14: "https://openai.com/index/video-generation-models-as-world-simulators/",
    15: "https://arxiv.org/abs/2501.00103",
    16: "https://arxiv.org/abs/2503.20314",
    17: "https://arxiv.org/abs/2407.01392",
    18: "https://arxiv.org/abs/2412.07772",
    19: "https://arxiv.org/abs/2506.08009",
    20: "https://arxiv.org/abs/2509.25161",
    21: "https://arxiv.org/abs/2505.13211",
    22: "https://arxiv.org/abs/2402.15391",
    23: "https://arxiv.org/abs/2501.03575",
    24: "https://arxiv.org/abs/2607.07534",
    25: "https://arxiv.org/abs/2605.15178",
    26: "https://github.com/lllyasviel/FramePack",
    27: "https://arxiv.org/abs/2504.12369",
    28: "https://arxiv.org/abs/2311.18828",
}


def cite(n: str) -> str:
    i = int(n)
    href = REF_URL.get(i, f"#ref-{i}")
    dest = href if href.startswith("http") else href
    extra = f' href="{dest}"' if dest.startswith("http") else f' href="#ref-{i}"'
    return f'<a class="cite" id="cite-{i}" href="#ref-{i}">{i}</a>'


def inline_format(text: str) -> str:
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
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", title.lower()).strip("-")
    return s or "section"


def md_to_html(md: str, inserts: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    lines = md.splitlines()
    out: list[str] = []
    toc: list[tuple[str, str]] = []
    i = 0
    in_code = False
    code: list[str] = []
    refs = False
    ref_items: list[str] = []

    def flush_code() -> None:
        nonlocal code
        if code:
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            code = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if line.strip() == "---":
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.strip() == "## References":
            refs = True
            out.append('<h2 id="references">References</h2>\n<div class="refs"><ol>')
            i += 1
            continue
        if refs:
            m = re.match(r"^(\d+)\.\s+(.*)$", line)
            if m:
                n, rest = m.group(1), m.group(2)
                rest = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', rest)
                rest = re.sub(r"\*(.+?)\*", r"<em>\1</em>", rest)
                ref_items.append(f'<li id="ref-{n}">{rest}</li>')
            elif line.startswith("    ") and ref_items:
                extra = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', line.strip())
                extra = re.sub(r"\*(.+?)\*", r"<em>\1</em>", extra)
                ref_items[-1] = ref_items[-1][:-5] + "<br>" + extra + "</li>"
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            sid = slugify(title)
            toc.append((sid, title))
            out.append(f'<h2 id="{sid}">{inline_format(html.escape(title))}</h2>')
            if sid in inserts:
                out.append(inserts[sid])
            i += 1
            continue
        if line.startswith("**") and line.endswith("**") is False and ". **" not in line:
            pass
        if not line.strip():
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
            out.append('<p class="math-block">\n' + "\n".join(block) + "\n</p>")
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].startswith("```") and lines[i].strip() != "---" and not lines[i].startswith("\\["):
            para.append(lines[i])
            i += 1
        text = " ".join(p.strip() for p in para)
        if text.startswith("**") and "**" in text[2:]:
            out.append("<p>" + inline_format(text) + "</p>")
        else:
            out.append("<p>" + inline_format(text) + "</p>")

    if ref_items:
        out.extend(ref_items)
        out.append("</ol></div>")
    return "\n".join(out), toc


HERO_EN = """
<div class="figure" aria-hidden="true">
  <div class="hero">
    <div class="tok-col">
      <span class="tok-up bad">mask at infer</span>
      <span class="stem"></span>
      <span class="tok">pretrain</span>
    </div>
    <div class="tok-col">
      <span class="tok">the product</span>
      <span class="stem"></span>
      <span class="tok-down ok">p(v<sub>t</sub> | v<sub>&lt;t</sub>)</span>
    </div>
    <div class="tok-col">
      <span class="tok-up warn">K-step ODE</span>
      <span class="stem"></span>
      <span class="tok">one factor</span>
    </div>
    <div class="tok-col">
      <span class="tok">one forward</span>
      <span class="stem"></span>
      <span class="tok-down ok">not a leftover integral</span>
    </div>
  </div>
  <p class="figure-caption">Native autoregression is the product written at pretraining. A causal mask, and a shorter ODE inside the block, are the usual substitutes.</p>
</div>
"""

HERO_ZH = """
<div class="figure" aria-hidden="true">
  <div class="hero">
    <div class="tok-col">
      <span class="tok-up bad">推理时加 mask</span>
      <span class="stem"></span>
      <span class="tok">预训练</span>
    </div>
    <div class="tok-col">
      <span class="tok">乘积</span>
      <span class="stem"></span>
      <span class="tok-down ok">p(v<sub>t</sub> | v<sub>&lt;t</sub>)</span>
    </div>
    <div class="tok-col">
      <span class="tok-up warn">块内 K 步 ODE</span>
      <span class="stem"></span>
      <span class="tok">一个因子</span>
    </div>
    <div class="tok-col">
      <span class="tok">一次前向</span>
      <span class="stem"></span>
      <span class="tok-down ok">不是一段剩余积分</span>
    </div>
  </div>
  <p class="figure-caption">原生自回归是预训练就写下的乘积。因果 mask，以及块内更短的微分方程，是现在常见的替代。</p>
</div>
"""

COMPARE_EN = """
<div class="pair">
  <div class="card bad">
    <div class="tag">short clip</div>
    <h3>Bidirectional teacher</h3>
    <p class="eq">\\(q_{\\mathrm{bi}}(v_t \\mid V_{\\setminus t}, c)\\)</p>
    <p>Future frames sit in the same example. Using them helps quality. The test that this wins is a few seconds, seen all at once.</p>
  </div>
  <div class="card ok">
    <div class="tag">stream</div>
    <h3>What a stream requires</h3>
    <p class="eq">\\(q_{\\mathrm{AR}}(v_t \\mid v_{&lt;t}, c_{\\le t})\\)</p>
    <p>Later frames have not occurred. A live \\(c_t\\) is not a license to see \\(v_{t+12}\\). Different visible variables, different function.</p>
  </div>
</div>
"""

COMPARE_ZH = """
<div class="pair">
  <div class="card bad">
    <div class="tag">短片段</div>
    <h3>双向 teacher</h3>
    <p class="eq">\\(q_{\\mathrm{bi}}(v_t \\mid V_{\\setminus t}, c)\\)</p>
    <p>未来帧就在同一条样本里，用上它们画质更好。它赢的测试，是一次看完的几秒钟。</p>
  </div>
  <div class="card ok">
    <div class="tag">流</div>
    <h3>流式测试要求的条件</h3>
    <p class="eq">\\(q_{\\mathrm{AR}}(v_t \\mid v_{&lt;t}, c_{\\le t})\\)</p>
    <p>后面的帧尚未发生。现场到达的 \\(c_t\\) 不是偷看 \\(v_{t+12}\\) 的许可。看见的变量不同，学到的函数就不同。</p>
  </div>
</div>
"""

PIPE_EN = """
<div class="pipeline">
  <div class="step"><b>pretrain</b><span>bidirectional DiT on a full short clip <span class="why">learns \\(q_{\\mathrm{bi}}\\)</span></span></div>
  <div class="step"><b>force</b><span>block-causal mask, clean history <span class="why">patches “past only”; history is still data</span></span></div>
  <div class="step"><b>short</b><span>self-rollout + DMD <span class="why">patches \\(K\\); history still short</span></span></div>
  <div class="step"><b>long</b><span>stitch + fixed window <span class="why">patches length</span></span></div>
  <div class="step"><b>search</b><span>window size, whether to reset positions <span class="why">patches \\(\\mathcal{H}_i\\); still not parameters</span></span></div>
</div>
<p class="figure-caption">The usual recipe, in the order the gaps show up. Each stage adds quantities to tune.</p>
"""

PIPE_ZH = """
<div class="pipeline">
  <div class="step"><b>预训练</b><span>双向 DiT，整段短片 <span class="why">学 \\(q_{\\mathrm{bi}}\\)</span></span></div>
  <div class="step"><b>强迫</b><span>块因果 mask，干净历史 <span class="why">补「只看过去」；历史仍是数据</span></span></div>
  <div class="step"><b>短程</b><span>自己滚出来的历史 + DMD <span class="why">补 \\(K\\)；历史仍短</span></span></div>
  <div class="step"><b>长程</b><span>拼接 + 固定窗口 <span class="why">补长度</span></span></div>
  <div class="step"><b>搜索</b><span>窗口多长、要不要重置位置 <span class="why">补 \\(\\mathcal{H}_i\\)；仍不是模型参数</span></span></div>
</div>
<p class="figure-caption">缺口按出现顺序一层层补上去。每一级多一组要调的量。</p>
"""

EXP_EN = """
<div class="pair">
  <div class="card">
    <div class="tag">probe 1 · weights</div>
    <h3>Later post-training</h3>
    <p>Same sampler, same window. Texture inside a block \\(B_i\\) gets cleaner. Consistency across blocks does not reliably follow.</p>
  </div>
  <div class="card">
    <div class="tag">probe 2 · \\(\\mathcal{H}_i\\)</div>
    <h3>Window and reset</h3>
    <p>Same generator. A longer window, or pose / position rebase on or off, changes who is visible at test time. Not \\(q_{\\mathrm{train}}\\).</p>
  </div>
</div>
"""

EXP_ZH = """
<div class="pair">
  <div class="card">
    <div class="tag">第一组 · 改权重</div>
    <h3>更晚的后训练</h3>
    <p>采样和窗口固定。同一块 \\(B_i\\) 里纹理更干净。跨块时间一致性没有稳定随之变好。</p>
  </div>
  <div class="card">
    <div class="tag">第二组 · \\(\\mathcal{H}_i\\)</div>
    <h3>窗口和重置</h3>
    <p>生成器固定。加长窗口，或开关位置 / 位姿重置，改的是测试时看见谁，不是训练条件。</p>
  </div>
</div>
"""

TABLE_EN = """
<table class="compare">
  <thead><tr><th></th><th>Current diffusion AR</th><th>Native</th></tr></thead>
  <tbody>
    <tr><td>Training condition</td><td>\\(q_{\\mathrm{bi}}\\)</td><td>\\(p(v_t \\mid v_{&lt;t}, c_{\\le t})\\)</td></tr>
    <tr><td>One factor</td><td>a \\(K\\)-step vector field</td><td>one forward</td></tr>
    <tr><td>Visible past \\(\\mathcal{H}_i\\)</td><td>searched at test time</td><td>structure in \\(\\theta\\)</td></tr>
    <tr><td>What later stages fix</td><td>mask, \\(K\\), window, rebase</td><td>nothing left to patch</td></tr>
  </tbody>
</table>
"""

TABLE_ZH = """
<table class="compare">
  <thead><tr><th></th><th>现行扩散自回归</th><th>原生</th></tr></thead>
  <tbody>
    <tr><td>训练条件</td><td>\\(q_{\\mathrm{bi}}\\)</td><td>\\(p(v_t \\mid v_{&lt;t}, c_{\\le t})\\)</td></tr>
    <tr><td>一个因子</td><td>\\(K\\) 步向量场</td><td>一次前向</td></tr>
    <tr><td>可见的过去 \\(\\mathcal{H}_i\\)</td><td>测试时搜索</td><td>\\(\\theta\\) 里的结构</td></tr>
    <tr><td>后面各阶段在补什么</td><td>mask、\\(K\\)、窗口、重置</td><td>没有对象可补</td></tr>
  </tbody>
</table>
"""

PAGES = {
    "en": {
        "src": SRC / "how_far_native_ar_video.md",
        "out": DOCS / "index.html",
        "title": "How Far We Are From a Native Autoregressive Video Model",
        "lang": "en",
        "other_href": "zh.html",
        "other_label": "中文",
        "self_label": "English",
        "date": "31 August 2026",
        "brand": "Notes",
        "toc_title": "Contents",
        "hero": HERO_EN,
        "inserts": {
            "an-image-has-no-time-it-must-obey-video-adds-t": COMPARE_EN,
            "training-sees-the-future-and-one-forward-does-not-emit-a-factor": TABLE_EN + PIPE_EN,
            "additional-experiments": EXP_EN,
        },
        "footer": 'Companion note to EchoWM. Source: <code>src/how_far_native_ar_video.md</code>.',
    },
    "zh": {
        "src": SRC / "how_far_native_ar_video.zh.md",
        "out": DOCS / "zh.html",
        "title": "我们离原生自回归视频模型还有多远",
        "lang": "zh-Hans",
        "other_href": "index.html",
        "other_label": "English",
        "self_label": "中文",
        "date": "2026年8月31日",
        "brand": "笔记",
        "toc_title": "目录",
        "hero": HERO_ZH,
        "inserts": {
            "图像里没有必须遵守的时间-视频多出来的是-t": COMPARE_ZH,
            "训练看见了未来-一次前向也写不出一个因子": TABLE_ZH + PIPE_ZH,
            "补充实验": EXP_ZH,
        },
        "footer": "EchoWM 的一篇配套笔记。文稿：<code>src/how_far_native_ar_video.zh.md</code>。",
    },
}


def page_html(cfg: dict, body: str, toc: list[tuple[str, str]]) -> str:
    toc_html = "\n".join(f'<a href="#{sid}">{html.escape(title)}</a>' for sid, title in toc)
    refs_label = "References" if cfg["lang"] == "en" else "参考文献"
    toc_html += f'\n<a href="#references">{refs_label}</a>'
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
  <header class="top">
    <a class="brand" href="index.html">{html.escape(cfg["brand"])}</a>
    <nav class="lang">
      <a href="index.html" class="{"is-on" if cfg["lang"] == "en" else ""}">English</a>
      <a href="zh.html" class="{"is-on" if cfg["lang"].startswith("zh") else ""}">中文</a>
    </nav>
  </header>
  <div class="layout">
    <nav class="toc">
      <div class="toc-title">{html.escape(cfg["toc_title"])}</div>
      {toc_html}
    </nav>
    <article>
      <p class="date">{html.escape(cfg["date"])}</p>
      <h1>{html.escape(cfg["title"])}</h1>
      {cfg["hero"]}
      {body}
      <footer class="note">{cfg["footer"]}</footer>
    </article>
  </div>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [
      {{left: '\\\\[', right: '\\\\]', display: true}},
      {{left: '\\\\(', right: '\\\\)', display: false}}
    ]}});"></script>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    for cfg in PAGES.values():
        md = cfg["src"].read_text(encoding="utf-8")
        body, toc = md_to_html(md, cfg["inserts"])
        cfg["out"].write_text(page_html(cfg, body, toc), encoding="utf-8")
        print("wrote", cfg["out"])


if __name__ == "__main__":
    main()
