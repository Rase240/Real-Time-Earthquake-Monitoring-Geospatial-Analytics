"""
Build docs/learning-guide.pdf from docs/learning-guide.md.

    pip install -r requirements-docs.txt
    python docs/build_guide.py            # writes docs/learning-guide.pdf
    python docs/build_guide.py --check    # only verify code excerpts against the sources

Pipeline
  1. Verify every code block preceded by `<!-- src: path -->` against that file:
     each run of lines (runs are separated by a line containing only "...") must
     appear verbatim and contiguously in the source. The build fails otherwise,
     so the guide cannot silently drift from the code.
  2. Markdown -> HTML (python-markdown + Pygments highlighting).
  3. HTML -> PDF with headless Chrome/Edge (two passes: the first finds the page
     of every heading, the second prints those numbers in the table of contents).
  4. PyMuPDF stamps running headers/footers, sets bookmarks and metadata, and
     renders PNG previews of selected pages into docs/build/ for inspection.
"""
import argparse
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
import pymupdf

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
SRC_MD = DOCS / "learning-guide.md"
OUT_PDF = DOCS / "learning-guide.pdf"
BUILD = DOCS / "build"
TITLE = "TerraWatcher — Real-Time Earthquake Analytics & Geospatial Intelligence"

SRC_RE = re.compile(r"<!-- src: (?P<path>[^ ]+) -->\n```(?P<lang>\w*)\n(?P<code>.*?)\n```", re.S)


# -----------------------------------------------------------------------------
# 1. Excerpt verification
# -----------------------------------------------------------------------------
def verify_excerpts(text):
    problems, count = [], 0
    for m in SRC_RE.finditer(text):
        count += 1
        path = ROOT / m["path"]
        if not path.exists():
            problems.append(f"{m['path']}: file not found")
            continue
        source = [ln.rstrip() for ln in path.read_text(encoding="utf-8").splitlines()]
        runs, current = [], []
        for line in m["code"].splitlines():
            if line.strip() in ("...", "# ...", "// ..."):
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(line.rstrip())
        if current:
            runs.append(current)
        start = 0
        for run in runs:
            found = next((i for i in range(start, len(source) - len(run) + 1)
                          if source[i:i + len(run)] == run), None)
            if found is None:
                problems.append(f"{m['path']}: excerpt not found verbatim (in order):\n    " + "\n    ".join(run[:4]))
                break
            start = found + len(run)
    return count, problems


# -----------------------------------------------------------------------------
# 2. Markdown -> HTML
# -----------------------------------------------------------------------------
def slug(value, sep="-"):
    value = re.sub(r"[^\w\s-]", "", value.lower()).strip()
    return re.sub(r"[\s_-]+", sep, value)


def to_html(text, toc_pages=None):
    # Code-source labels.
    def label(m):
        return f'<div class="src-label">{html.escape(m["path"])}</div>\n\n```{m["lang"]}\n{m["code"]}\n```'
    text = SRC_RE.sub(label, text)
    md = markdown.Markdown(
        extensions=["extra", "codehilite", "toc", "sane_lists", "smarty"],
        extension_configs={
            "codehilite": {"css_class": "hl", "guess_lang": False},
            "toc": {"slugify": slug, "toc_depth": "1-2"},
            "smarty": {"smart_angled_quotes": False},
        },
    )
    body = md.convert(text)
    body = body.replace('src="images/', f'src="{(DOCS / "images").as_uri()}/')
    body = body.replace('src="diagrams/', f'src="{(DOCS / "diagrams").as_uri()}/')

    # Table of contents from h1 (chapters) and h2 (sections), with page numbers once known.
    rows = []
    for h1 in md.toc_tokens:
        if h1["id"] in ("contents",):
            continue
        rows.append(("chapter", h1))
        rows += [("section", h2) for h2 in h1.get("children", [])]
    toc = ['<nav class="toc"><h1 class="toc-title" id="contents">Contents</h1><ol>']
    for kind, tok in rows:
        page = (toc_pages or {}).get(tok["id"], "")
        name = html.unescape(re.sub(r"<[^>]+>", "", tok["name"]))
        toc.append(f'<li class="{kind}"><a href="#{tok["id"]}"><span class="t">{html.escape(name)}</span>'
                   f'<span class="dots"></span><span class="pg">{page}</span></a></li>')
    toc.append("</ol></nav>")
    headings = [(kind, tok["id"], html.unescape(re.sub(r"<[^>]+>", "", tok["name"]))) for kind, tok in rows]
    return body, "\n".join(toc), headings


def page_html(body, toc):
    css = (DOCS / "guide.css").read_text(encoding="utf-8")
    from pygments.formatters import HtmlFormatter
    css += HtmlFormatter(style="friendly").get_style_defs(".hl")
    cover = f"""
<section class="cover">
  <div class="cover-grid"></div>
  <div class="cover-inner">
    <div class="cover-kicker">Technical learning guide</div>
    <h1 class="cover-title">TerraWatcher</h1>
    <div class="cover-sub">Real-Time Earthquake Analytics &amp; Geospatial Intelligence</div>
    <img class="cover-img" src="{(DOCS / 'images' / 'dashboard-desktop.png').as_uri()}" alt="TerraWatcher dashboard">
    <div class="cover-meta">
      Architecture · Data pipeline · Geospatial concepts · Machine learning · Scientific validity ·
      Deployment · Testing · Interview preparation
    </div>
    <div class="cover-foot">Written against the repository state of 19 October 2026 · Flask · Leaflet · PyTorch · USGS FDSN</div>
  </div>
</section>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(TITLE)} — Learning Guide</title>
<style>{css}</style></head><body>{cover}{toc}<main>{body}</main></body></html>"""


# -----------------------------------------------------------------------------
# 3. PDF rendering
# -----------------------------------------------------------------------------
def find_browser():
    candidates = [
        os.environ.get("CHROME_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("Chrome/Chromium/Edge not found; set CHROME_PATH")


def render_pdf(html_text, pdf_path):
    BUILD.mkdir(exist_ok=True)
    html_path = BUILD / "learning-guide.html"
    html_path.write_text(html_text, encoding="utf-8")
    profile = BUILD / "chrome-profile"
    cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
           f"--user-data-dir={profile}", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=10000", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not Path(pdf_path).exists():
        sys.exit("Browser did not produce a PDF")


def heading_pages(pdf_path, headings):
    """Map heading id -> 1-based page number using the TOC's internal links."""
    doc = pymupdf.open(pdf_path)
    links = []
    for pno in range(min(6, len(doc))):
        for link in doc[pno].get_links():
            if link.get("kind") == pymupdf.LINK_GOTO:
                links.append(link["page"] + 1)
            elif link.get("kind") == pymupdf.LINK_NAMED and "page" in link:
                links.append(link["page"] + 1)
    doc.close()
    if len(links) < len(headings):
        sys.exit(f"Expected {len(headings)} TOC links, found {len(links)}")
    return {hid: page for (_, hid, _), page in zip(headings, links[:len(headings)], strict=True)}


def finish(pdf_path, out_path, headings, pages):
    doc = pymupdf.open(pdf_path)
    chapters = sorted(((pages[h], name) for kind, h, name in headings if kind == "chapter"))
    toc_start = 2
    for pno in range(len(doc)):
        n = pno + 1
        if n == 1:
            continue
        page = doc[pno]
        w, h = page.rect.width, page.rect.height
        current = next((name for start, name in reversed(chapters) if start <= n), "Contents")
        if n >= toc_start:
            page.insert_text((48, 30), "TerraWatcher · Learning Guide", fontsize=7.5, fontname="helv", color=(0.42, 0.5, 0.56))
            label = current if len(current) < 70 else current[:67] + "…"
            tw = pymupdf.get_text_length(label, fontname="helv", fontsize=7.5)
            page.insert_text((w - 48 - tw, 30), label, fontsize=7.5, fontname="helv", color=(0.42, 0.5, 0.56))
            page.draw_line((48, 36), (w - 48, 36), color=(0.85, 0.89, 0.91), width=0.5)
            num = str(n)
            nw = pymupdf.get_text_length(num, fontname="helv", fontsize=8.5)
            page.insert_text(((w - nw) / 2, h - 22), num, fontsize=8.5, fontname="helv", color=(0.3, 0.38, 0.44))
    outline = [[1, "Cover", 1], [1, "Contents", toc_start]]
    for kind, hid, name in headings:
        outline.append([1 if kind == "chapter" else 2, name, pages[hid]])
    doc.set_toc(outline)
    doc.set_metadata({"title": TITLE + " — Technical Learning Guide", "author": "TerraWatcher project",
                      "subject": "Architecture, data pipeline, ML and evaluation of TerraWatcher",
                      "keywords": "earthquake, USGS, Flask, Leaflet, PyTorch, geospatial"})
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()


def previews(pdf_path, pages):
    out = BUILD / "preview"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    doc = pymupdf.open(pdf_path)
    for n in pages:
        if 1 <= n <= len(doc):
            doc[n - 1].get_pixmap(dpi=70).save(out / f"page-{n:02d}.png")
    total = len(doc)
    doc.close()
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify excerpts")
    ap.add_argument("--preview", default="1,2,3,5,8,12,20,30,40", help="comma-separated pages to render as PNG")
    args = ap.parse_args()

    text = SRC_MD.read_text(encoding="utf-8")
    count, problems = verify_excerpts(text)
    print(f"Verified {count} source excerpts")
    if problems:
        print("\n".join(problems))
        return 1
    if args.check:
        return 0

    body, toc, headings = to_html(text)
    pass1 = BUILD / "pass1.pdf"
    render_pdf(page_html(body, toc), pass1)
    pages = heading_pages(pass1, headings)

    body, toc, _ = to_html(text, pages)
    pass2 = BUILD / "pass2.pdf"
    render_pdf(page_html(body, toc), pass2)
    if heading_pages(pass2, headings) != pages:
        print("Warning: page numbers shifted between passes")
    finish(pass2, OUT_PDF, headings, pages)
    out, total = previews(OUT_PDF, [int(p) for p in args.preview.split(",") if p])
    print(f"Wrote {OUT_PDF.relative_to(ROOT)} ({total} pages); previews in {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
