#!/usr/bin/env python3
"""Build A4 PDFs from the essay. Chinese pages are inter-character justified and compact."""

from __future__ import annotations

import html
import re
import shutil
import urllib.request
from pathlib import Path

from build_site import md_to_html

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
FONTS = ROOT / "fonts"
BUILD = ROOT / "_pdf_build"
PDF_DIR = ROOT / "pdf"

FONT_FILES = {
    "NotoSerifCJKsc-Regular.otf": (
        "https://github.com/notofonts/noto-cjk/raw/Serif2.003/"
        "Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf"
    ),
    "NotoSerifCJKsc-Bold.otf": (
        "https://github.com/notofonts/noto-cjk/raw/Serif2.003/"
        "Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf"
    ),
}

PAGES = {
    "en": {
        "src": SRC / "how_far_native_ar_video.md",
        "stem": "how-far-native-ar-video.en",
        "title": "How Far Are We from a Native Autoregressive Video Model?",
        "lang": "en",
        "date": "31 August 2026",
        "byline": "Weiyang Jin, The University of Hong Kong",
        "cite_h": "Cite",
        "credit": (
            "Visual style after "
            '<a href="https://thinkingmachines.ai/blog/on-policy-distillation/">Thinking Machines Lab</a>.'
        ),
        "head": "How Far Are We from a Native Autoregressive Video Model?",
    },
    "zh": {
        "src": SRC / "how_far_native_ar_video.zh.md",
        "stem": "how-far-native-ar-video.zh",
        "title": "我们离原生自回归视频模型还有多远",
        "lang": "zh-Hans",
        "date": "2026年8月31日",
        "byline": "Weiyang Jin，香港大学",
        "cite_h": "引用",
        "credit": (
            "版式参考了 "
            '<a href="https://thinkingmachines.ai/blog/on-policy-distillation/">Thinking Machines Lab</a> '
            "的研究笔记，谨此致谢。"
        ),
        "head": "我们离原生自回归视频模型还有多远",
    },
}

BIBTEX = """@misc{jin2026nativear,
  author       = {Jin, Weiyang},
  title        = {How Far Are We from a Native Autoregressive Video Model?},
  year         = {2026},
  month        = aug,
  institution  = {The University of Hong Kong},
  howpublished = {Blog post},
  url          = {https://waynejin0918.github.io/how-far-native-ar-video/},
}"""

SEE_FIG = re.compile(r'<figure class="fig fig-see".*?</figure>', re.S)


def ensure_fonts() -> None:
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_FILES.items():
        dest = FONTS / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            continue
        print("download", name)
        req = urllib.request.Request(url, headers={"User-Agent": "how-far-native-ar-video-pdf"})
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out)


def frame_row(mode: str) -> str:
    cells = []
    for i in range(1, 9):
        if i < 5:
            kind = "past"
        elif i == 5:
            kind = "now"
        else:
            kind = "past" if mode == "bi" else "future"
        cells.append(f'<div class="frame {kind}"><span>{i}</span></div>')
    return (
        '<div class="see-axis"><div class="lab">t</div>'
        f'<div class="frames">{"".join(cells)}</div></div>'
    )


def print_see(zh: bool) -> str:
    if zh:
        rows = (
            ("q_{\\mathrm{bi}}", "bi", "短片训练里写第 5 帧，1–8 都可以进条件。"),
            ("q_{\\mathrm{AR}}", "ar", "流式生成只看见已经发生的。写第 5 帧时，6–8 还不存在。"),
        )
    else:
        rows = (
            ("q_{\\mathrm{bi}}", "bi", "On a short clip, writing frame 5, 1–8 are all legal."),
            ("q_{\\mathrm{AR}}", "ar", "Streaming generation only admits the past. Writing frame 5, 6–8 do not exist."),
        )
    parts = ['<figure class="fig fig-see print-see">']
    for lab, mode, note in rows:
        parts.append(
            f'<div class="print-see-row" data-mode="{mode}">'
            f'<p class="lab-row">\\({lab}\\)</p>'
            f"{frame_row(mode)}"
            f'<p class="see-note">{html.escape(note)}</p>'
            "</div>"
        )
    parts.append("</figure>")
    return "\n".join(parts)


def print_html(cfg: dict, body: str) -> str:
    see = print_see(cfg["lang"].startswith("zh"))
    body = SEE_FIG.sub(lambda _: see, body, count=1)
    body = body.replace('<ol class="stack">', '<ol class="stack print-open">')
    body = re.sub(r'<li(?: class="on")?>', "<li class=\"on\">", body)
    regular = (FONTS / "NotoSerifCJKsc-Regular.otf").resolve().as_uri()
    bold = (FONTS / "NotoSerifCJKsc-Bold.otf").resolve().as_uri()
    return f"""<!DOCTYPE html>
<html lang="{cfg["lang"]}">
<head>
  <meta charset="utf-8">
  <title>{html.escape(cfg["title"])}</title>
  <style>
    @font-face {{
      font-family: "Noto Serif SC";
      src: url("{regular}") format("opentype");
      font-weight: 400;
    }}
    @font-face {{
      font-family: "Noto Serif SC";
      src: url("{bold}") format("opentype");
      font-weight: 600;
    }}
  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <link rel="stylesheet" href="{(ROOT / "css/blog.css").resolve().as_uri()}">
  <link rel="stylesheet" href="{(ROOT / "css/print.css").resolve().as_uri()}">
</head>
<body class="print">
  <main class="page">
    <article>
      <h1>{html.escape(cfg["title"])}</h1>
      <p class="byline">{html.escape(cfg["byline"])} · {html.escape(cfg["date"])}</p>
      {body}
      <h2 id="cite">{html.escape(cfg["cite_h"])}</h2>
      <div class="bib"><pre><code>{html.escape(BIBTEX)}</code></pre></div>
    </article>
    <footer class="credit">{cfg["credit"]}</footer>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script>
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: "\\\\[", right: "\\\\]", display: true}},
        {{left: "\\\\(", right: "\\\\)", display: false}}
      ],
      throwOnError: false,
      strict: false
    }});
    document.documentElement.dataset.ready = "1";
  </script>
</body>
</html>
"""


def write_print_html(key: str) -> Path:
    cfg = PAGES[key]
    body = md_to_html(cfg["src"].read_text(encoding="utf-8"), zh=key == "zh")
    BUILD.mkdir(exist_ok=True)
    path = BUILD / (cfg["stem"] + ".html")
    path.write_text(print_html(cfg, body), encoding="utf-8")
    return path


def html_to_pdf(html_path: Path, pdf_path: Path, header: str) -> None:
    from playwright.sync_api import sync_playwright

    PDF_DIR.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=120000)
        page.wait_for_function("() => document.documentElement.dataset.ready === '1'", timeout=60000)
        page.evaluate("() => document.fonts.ready")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=(
                '<div style="font-size:8.5px;color:#6e6658;width:100%;'
                'padding:0 15mm;font-family:serif;">'
                f"{html.escape(header)}</div>"
            ),
            footer_template=(
                '<div style="font-size:8.5px;color:#6e6658;width:100%;'
                'text-align:center;font-family:serif;">'
                '<span class="pageNumber"></span></div>'
            ),
            margin={"top": "13mm", "bottom": "12mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()
    print("wrote", pdf_path)


def build_one(key: str) -> Path:
    html_path = write_print_html(key)
    pdf_path = PDF_DIR / (PAGES[key]["stem"] + ".pdf")
    html_to_pdf(html_path, pdf_path, PAGES[key]["head"])
    return pdf_path


def main() -> None:
    ensure_fonts()
    for key in ("zh", "en"):
        build_one(key)


if __name__ == "__main__":
    main()
