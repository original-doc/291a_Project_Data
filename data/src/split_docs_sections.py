#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split each JSON item into multiple items by headings that start with '## ' or '### ',
then filter out short sections by min words / min chars.

Rules:
- A section starts at a line that matches either:
    '## <title>'  or  '### <title>'
  (leading spaces allowed).
- Section body spans until the next '##'/'###' or EOF.
- Only sections with '##' or '###' are emitted; content before the first such
  heading is ignored.
- Child item:
    - inherits all parent metadata except 'title'/'text'/'index'
    - sets 'title' to the section heading text
    - sets 'text' to the section body (without the heading line)
    - adds 'parent_index' = parent's 'index'
- Finally re-enumerate 'index' globally starting at 0.

Filtering:
- Use --min-words and/or --min-chars (both default 0 = disabled).
- If both are provided, a section must satisfy both thresholds.

Usage:
    python src/split_docs_sections.py --in ./final_data/docs.json --out ./final_data/docs.splitted.json
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

HEAD_RE = re.compile(r'^\s*(##|###)\s+(.*\S)\s*$')

def extract_sections(text: str) -> List[Tuple[str, str]]:
    """Return a list of (title, body) for each '##' or '###' section."""
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []

    cur_title: str = ""
    cur_buf: List[str] = []
    in_section = False

    def flush():
        nonlocal cur_title, cur_buf, in_section
        if in_section:
            body = "\n".join(cur_buf).rstrip("\n").lstrip("\n")
            sections.append((cur_title, body))
        cur_title = ""
        cur_buf = []
        in_section = False

    for ln in lines:
        m = HEAD_RE.match(ln)
        if m:
            # new section begins; flush previous
            flush()
            cur_title = m.group(2).strip()
            in_section = True
            cur_buf = []
        else:
            if in_section:
                cur_buf.append(ln)

    # tail
    flush()
    return sections

def count_words(s: str) -> int:
    """Count words by whitespace splitting."""
    return len([w for w in s.strip().split() if w])

def passes_thresholds(body: str, min_words: int, min_chars: int) -> bool:
    """Check whether the body satisfies the thresholds."""
    if min_words <= 0 and min_chars <= 0:
        return True
    if min_words > 0 and count_words(body) < min_words:
        return False
    if min_chars > 0 and len(body.strip()) < min_chars:
        return False
    return True

def main():
    ap = argparse.ArgumentParser(description="Split JSON items by '##'/'###' and filter short sections.")
    ap.add_argument("--in", dest="inp,",
                    required=True, help="Input JSON path (array of objects).")
    ap.add_argument("--out", dest="out",
                    required=True, help="Output JSON path.")
    ap.add_argument("--min-words", dest="min_words", type=int, default=10,
                    help="Minimum word count to keep a section (default 0 = disabled).")
    ap.add_argument("--min-chars", dest="min_chars", type=int, default=0,
                    help="Minimum character count to keep a section (default 0 = disabled).")
    args = ap.parse_args()

    # Work around comma in dest due to '--in' naming
    inp_path = getattr(args, "inp,")
    out_path = args.out

    inp = Path(inp_path)
    outp = Path(out_path)

    try:
        data = json.loads(inp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Input JSON must be an array.")
    except Exception as e:
        raise SystemExit(f"Failed to read input JSON: {e}")

    results: List[Dict[str, Any]] = []

    for parent in data:
        parent_text = parent.get("text") or ""
        sections = extract_sections(parent_text)
        if not sections:
            continue

        parent_index = parent.get("index")

        for title, body in sections:
            if not passes_thresholds(body, args.min_words, args.min_chars):
                continue
            child = {k: v for k, v in parent.items() if k not in ("title", "text", "index")}
            child["title"] = title
            child["text"] = body
            child["parent_index"] = parent_index
            results.append(child)

    # Re-enumerate global index
    for idx, obj in enumerate(results):
        obj["index"] = idx

    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Split {len(data)} parents -> kept {len(results)} children -> {outp} (min_words={args.min_words}, min_chars={args.min_chars})")

if __name__ == "__main__":
    main()
