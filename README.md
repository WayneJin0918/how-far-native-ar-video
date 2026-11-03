# How Far Are We from a Native Autoregressive Video Model?

Bilingual note. Pages:

- English: [index.html](index.html)
- 中文: [zh.html](zh.html)

Edit `src/*.md`, then:

```bash
python3 build_site.py
python3 build_pdf.py
```

PDFs land in `pdf/`. Chinese pages are inter-character justified and set compact so each A4 page fills. The first run downloads Noto Serif CJK into `fonts/` and needs Playwright Chromium (`python3 -m pip install playwright && python3 -m playwright install chromium`).

GitHub Pages is deployed from the repository root by `.github/workflows/pages.yml`. After the first push, enable **Settings → Pages → GitHub Actions**.
