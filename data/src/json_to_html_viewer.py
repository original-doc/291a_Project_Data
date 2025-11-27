#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn a JSON array (each item has fields like: title, text, url_html, index)
into a single HTML file for quick visual browsing and keyword search.

Usage:
    python src/json_to_html_viewer.py --in final_data/docs.json --out viewer.html
"""
import argparse
import json
import sys
from pathlib import Path
from html import escape

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>JSON Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { --bg:#0b0d12; --fg:#e7e9ee; --muted:#aab1bd; --card:#161a22; --accent:#4da3ff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji", sans-serif; background:var(--bg); color:var(--fg); }
  header { position:sticky; top:0; z-index:10; background:rgba(11,13,18,0.92); backdrop-filter:saturate(1.2) blur(6px); border-bottom:1px solid #222835; }
  .container { max-width: 1200px; margin: 0 auto; padding: 16px; }
  .toolbar { display:flex; gap:12px; align-items:center; }
  .toolbar input[type="search"] { flex:1; padding:10px 12px; border-radius:10px; border:1px solid #2a3140; background:#0e121a; color:var(--fg); outline:none; }
  .toolbar input[type="search"]::placeholder { color:#79808d; }
  .count { font-size: 13px; color: var(--muted); white-space:nowrap; }
  main { display:grid; grid-template-columns: 270px 1fr; gap: 16px; padding:16px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } .toc { position: static; max-height: none; } }
  .toc { position: sticky; top:70px; align-self:start; max-height: calc(100vh - 90px); overflow:auto; background: var(--card); border:1px solid #202636; border-radius: 12px; padding: 12px; }
  .toc h2 { margin: 4px 0 8px; font-size:14px; color: var(--muted); font-weight:600; }
  .toc a { display:block; color:#cfd6e3; text-decoration:none; padding:6px 8px; border-radius:8px; }
  .toc a:hover { background:#1c2230; }
  .toc a.active { background:#1f2737; color:#fff; border-left:3px solid var(--accent); padding-left:5px; }
  .entry { background: var(--card); border:1px solid #202636; border-radius: 12px; margin-bottom: 18px; }
  .entry header { position: sticky; top: 70px; background: #151924; border-bottom:1px solid #202636; border-radius: 12px 12px 0 0; }
  .entry-header { display:flex; flex-wrap:wrap; gap:8px 12px; align-items: center; justify-content: space-between; padding: 10px 12px; }
  .meta { font-size: 12px; color: var(--muted); display:flex; gap:10px; align-items:center; }
  .meta a { color: var(--accent); text-decoration: none; }
  .navbtns { display:flex; gap:8px; }
  .btn { padding:6px 10px; border-radius: 8px; border:1px solid #263044; background:#111726; color:#dfe6f3; text-decoration:none; cursor:pointer; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .title { margin:0; font-weight:700; font-size:16px; }
  .content { padding: 12px; }
  pre { margin:0; white-space: pre-wrap; word-wrap: break-word; font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color:#e8eaf2; }
  mark { background:#3a4a6b; color:#fff; padding:0 2px; border-radius:3px; }
  .footer-note { text-align:center; color: var(--muted); padding: 16px; font-size:12px; }
</style>
</head>
<body>
<header>
  <div class="container toolbar">
    <input id="q" type="search" placeholder="Type to filter (title + text). Press / to focus, Esc to clear." />
    <div class="count" id="count"></div>
  </div>
</header>

<main class="container">
  <nav class="toc" id="toc">
    <h2>Entries</h2>
    {TOC}
  </nav>

  <section id="entries">
    {ENTRIES}
  </section>
</main>

<div class="footer-note">Keyboard: j = next, k = prev, / = focus search, Enter = go to first match, Esc = clear search</div>

<script>
(function(){
  const q = document.getElementById('q');
  const count = document.getElementById('count');
  const entries = Array.from(document.querySelectorAll('.entry'));
  const tocLinks = Array.from(document.querySelectorAll('.toc a'));

  function normalize(s){ return (s||'').toLowerCase(); }

  function updateCount() {
    const visible = entries.filter(e => !e.classList.contains('hidden')).length;
    count.textContent = visible + ' / ' + entries.length;
  }

  function clearMarks(node) {
    node.querySelectorAll('mark').forEach(m => {
      const parent = m.parentNode;
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    });
  }

  function highlight(node, term) {
    if (!term) return;
    const walk = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null);
    const texts = [];
    while (walk.nextNode()) texts.push(walk.currentNode);
    const re = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    texts.forEach(t => {
      if (!t.nodeValue.trim()) return;
      const frag = document.createDocumentFragment();
      let lastIndex = 0;
      t.nodeValue.replace(re, (match, idx) => {
        frag.appendChild(document.createTextNode(t.nodeValue.slice(lastIndex, idx)));
        const m = document.createElement('mark');
        m.textContent = match;
        frag.appendChild(m);
        lastIndex = idx + match.length;
        return match;
      });
      if (lastIndex === 0) return;
      frag.appendChild(document.createTextNode(t.nodeValue.slice(lastIndex)));
      t.parentNode.replaceChild(frag, t);
    });
  }

  function filter() {
    const term = normalize(q.value);
    entries.forEach(e => {
      e.classList.remove('hidden');
      clearMarks(e);
    });
    tocLinks.forEach(a => a.classList.remove('active'));

    if (!term) { updateCount(); return; }

    let firstMatch = null;
    entries.forEach(e => {
      const hay = normalize(e.dataset.search);
      if (!hay.includes(term)) {
        e.classList.add('hidden');
      } else {
        if (!firstMatch) firstMatch = e;
        highlight(e.querySelector('.content'), term);
      }
    });
    updateCount();
    if (firstMatch) {
      document.querySelector('.toc a[href="#' + firstMatch.id + '"]')?.classList.add('active');
    }
  }

  q.addEventListener('input', filter);
  document.addEventListener('keydown', (ev) => {
    if (ev.key === '/') { ev.preventDefault(); q.focus(); q.select(); }
    if (ev.key === 'Escape') { q.value=''; filter(); }
    if (ev.key === 'Enter') {
      const first = entries.find(e => !e.classList.contains('hidden'));
      if (first) location.hash = '#' + first.id;
    }
    if (ev.key === 'j' || ev.key === 'k') {
      ev.preventDefault();
      const visible = entries.filter(e => !e.classList.contains('hidden'));
      const idx = visible.findIndex(e => ('#'+e.id) === location.hash);
      let target = null;
      if (ev.key === 'j') target = visible[Math.min((idx<0?0:idx+1), visible.length-1)];
      if (ev.key === 'k') target = visible[Math.max((idx<0?0:idx-1), 0)];
      if (target) location.hash = '#' + target.id;
    }
  });

  // sync active in TOC
  window.addEventListener('hashchange', () => {
    tocLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === location.hash));
  });

  // initial
  updateCount();
})();
</script>
</body>
</html>
"""

def build_html(items):
    # Build TOC
    toc_parts = []
    entry_parts = []
    for it in items:
        idx = it.get("index")
        idx_disp = str(idx) if idx is not None else ""
        title = it.get("title") or ""
        url_html = it.get("url_html") or ""
        text = it.get("text") or ""

        # Escape everything to show "raw source" text safely
        title_esc = escape(title)
        text_esc = escape(text)

        # Anchor id
        anchor = f"entry-{idx_disp}" if idx_disp != "" else f"entry-{len(entry_parts)}"

        # TOC item
        toc_parts.append(
            f'<a href="#{anchor}" title="Go to {title_esc}">[{idx_disp}] {title_esc or "(no title)"}'
            f'</a>'
        )

        # Meta (index + source link)
        meta_bits = []
        if idx_disp != "":
            meta_bits.append(f"index: {idx_disp}")
        if url_html:
            meta_bits.append(f'<a href="{escape(url_html)}" target="_blank" rel="noreferrer">source</a>')
        meta_html = " · ".join(meta_bits) if meta_bits else "&nbsp;"

        # One entry card
        entry_parts.append(
            f'''<article class="entry" id="{anchor}" data-search="{escape((title or "") + " " + (text or ""))}">
  <header>
    <div class="entry-header">
      <h3 class="title">{title_esc or "(no title)"}</h3>
      <div class="meta">{meta_html}</div>
      <div class="navbtns">
        <a class="btn" href="#entry-{(int(idx_disp)-1) if idx_disp.isdigit() and int(idx_disp)>0 else 0}">Prev</a>
        <a class="btn" href="#entry-{(int(idx_disp)+1) if idx_disp.isdigit() else f"entry-{len(entry_parts)+1}"}">Next</a>
      </div>
    </div>
  </header>
  <div class="content">
    <pre>{text_esc}</pre>
  </div>
</article>'''
        )

    html = HTML_TEMPLATE.replace("{TOC}", "\n".join(toc_parts)).replace("{ENTRIES}", "\n".join(entry_parts))
    return html

def main():
    ap = argparse.ArgumentParser(description="Render JSON items (title/text/url_html/index) into a browsable HTML file.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON path (array of objects).")
    ap.add_argument("--out", dest="out", required=True, help="Output HTML path.")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)

    try:
        data = json.loads(inp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print("ERROR: Input JSON must be an array of objects.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR reading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    html = build_html(data)
    outp.write_text(html, encoding="utf-8")
    print(f"Done -> {outp}")

if __name__ == "__main__":
    main()
