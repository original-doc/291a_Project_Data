#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean JSONL items containing RST (`rst_text`) and output a single JSON array.
Incremental changes:
  - Strip standalone ':orphan:' lines.
  - Sanitize `title` by removing non-ASCII chars (e.g., '¶').
  - Wrap ALL code blocks in Markdown fences in `text`:
      * reStructuredText code directives: '.. code-block:: <lang>' and '.. code::'
      * literal blocks introduced by paragraphs ending with '::'
  - Convert reStructuredText section titles (overline/underline styles) to Markdown '#'-style headings.
  - Prior behavior kept: field slimming and optional saving of `code_blocks`.

python src/clean_docs_rst_jsonl.py --in .\unfiltered_data\lightning_docs_rst.jsonl   --out .\final_data\lightning_docs_cleaned.json
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Default directive names to strip entirely (remove the directive line + its indented block).
DEFAULT_STRIP_DIRECTIVES = {
    "video", "image", "figure", "raw", "include", "toctree",
    "embed", "gallery", "thumbnail", "plot", "testsetup", "testcleanup",
    # add more if you encounter noisy directives
}

# Regexes for inline cleanup
RE_INLINE_ROLE = re.compile(r":\w+:`([^`]+)`")          # :class:`Foo` -> Foo
RE_ANGLE_LINK  = re.compile(r"`([^`<]+?)\s*<[^>`]+>`_") # `text <url>`_ -> text
RE_BARE_LINK   = re.compile(r"`([^`]+?)`_")             # `something`_ -> something
RE_LITERAL     = re.compile(r"``([^`]+)``")             # ``code`` -> code
RE_SUBST       = re.compile(r"\|[^|\s]+\|")             # |subst| -> (drop)
RE_WS          = re.compile(r"\s+")

# Heading adornment precedence for Markdown level mapping (left -> higher level)
HEADING_ORDER = ['=', '#', '^', '"', '+', '-', ':', '.']

def strip_orphan_lines(text: str) -> str:
    """Remove standalone ':orphan:' and hyperlink target lines like '.. _label:'."""
    lines = text.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        # drop orphan
        if s == ":orphan:":
            continue
        # drop hyperlink target like ".. _something:"
        if re.match(r"^\.\.\s*_[A-Za-z0-9_.-]+:\s*$", s):
            continue
        kept.append(ln)
    return "\n".join(kept)

def sanitize_title(title: str) -> str:
    """Remove non-ASCII chars (e.g., pilcrow '¶') and trim."""
    if not title:
        return ""
    # Drop all non-ASCII characters
    t = title.encode("ascii", "ignore").decode("ascii")
    # Collapse whitespace
    t = RE_WS.sub(" ", t).strip()
    return t

def dedent_block(lines: List[str], start: int, base_indent: int) -> Tuple[str, int]:
    """Collect an indented block after a directive or paragraph.
    Returns (block_text, next_index). The block ends when indentation <= base_indent or at EOF.
    """
    collected: List[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            collected.append("")  # keep paragraph breaks
            i += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= base_indent:
            break
        # Normalize a single indentation level relative to the base
        to_strip = min(indent - base_indent, 4)  # typical directive blocks indent by ~3
        collected.append(line[to_strip:])
        i += 1
    text = "\n".join(collected).rstrip()
    return text, i

def convert_headings_rst_to_md(lines: List[str]) -> List[str]:
    """Detect RST section titles (overline+underline or underline-only) and convert to Markdown headings."""
    out: List[str] = []
    i = 0
    def adorn_level(ch: str) -> int:
        # Map adornment char to Markdown level (1..6); fallback to 3
        if ch in HEADING_ORDER:
            idx = HEADING_ORDER.index(ch)
            return min(idx + 1, 6)
        return 3

    while i < len(lines):
        line = lines[i]
        # Pattern A: overline + title + underline (same char)
        if i + 2 < len(lines):
            over = lines[i].rstrip()
            title = lines[i+1].strip()
            under = lines[i+2].rstrip()
            if over and under and title and set(over) == {over[0]} and set(under) == {under[0]} and over[0] == under[0]:
                if len(over) >= len(title) and len(under) >= len(title):
                    lvl = adorn_level(over[0])
                    out.append("#" * lvl + " " + title)
                    i += 3
                    continue
        # Pattern B: title + underline (same char repeated)
        if i + 1 < len(lines):
            title = lines[i].strip()
            under = lines[i+1].rstrip()
            if title and under and set(under) == {under[0]} and len(under) >= len(title):
                lvl = adorn_level(under[0])
                out.append("#" * lvl + " " + title)
                i += 2
                continue
        # Default
        out.append(line)
        i += 1
    return out

# 允许指令前有缩进
RE_DIRECTIVE = re.compile(r"^\s*\.\.\s+([a-zA-Z0-9_-]+)::(.*)$")

def _normalize_outside_code(text: str) -> str:
    """Run inline/link cleanup and whitespace collapsing only outside fenced code blocks."""
    lines = text.splitlines()
    out = []
    in_fence = False
    for ln in lines:
        stripped = ln.strip()
        # Detect fence lines: ``` or ```lang
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(ln)  # keep fence as-is
            continue
        if in_fence:
            out.append(ln)  # keep code lines untouched
            continue
        # Outside code blocks: apply inline cleanup
        ln2 = RE_INLINE_ROLE.sub(r"\1", ln)
        ln2 = RE_ANGLE_LINK.sub(r"\1", ln2)
        ln2 = RE_BARE_LINK.sub(r"\1", ln2)
        ln2 = RE_LITERAL.sub(r"\1", ln2)
        ln2 = RE_SUBST.sub("", ln2)
        # Collapse excessive whitespace but keep empty lines
        ln2 = RE_WS.sub(" ", ln2).strip()
        out.append(ln2)
    # Remove only *consecutive* empty lines outside code;
    # here we just collapse runs of empty lines while preserving single blanks
    cleaned = []
    blank_run = 0
    for ln in out:
        if not ln:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append(ln)
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned).strip()

def skip_directive_options(lines: List[str], start: int, base_indent: int) -> int:
    """Skip blank lines and option lines (indented and starting with ':'). Return index of body start."""
    j = start
    while j < len(lines):
        l2 = lines[j]
        if not l2.strip():
            j += 1
            continue
        indent2 = len(l2) - len(l2.lstrip(" "))
        if indent2 <= base_indent:
            break
        if l2.lstrip().startswith(":"):
            j += 1
            continue
        break
    return j

def parse_rst_keep_code(rst: str, strip_directives: set) -> Tuple[str, List[Dict[str, str]]]:
    """Remove selected directives, convert headings, and keep/wrap code blocks."""
    # 1) Pre-strip orphan lines
    rst = strip_orphan_lines(rst)
    # 2) Convert headings early
    orig_lines = rst.splitlines()
    lines = convert_headings_rst_to_md(orig_lines)

    out_text_lines: List[str] = []
    code_blocks: List[Dict[str, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # A. Any directive (with possible leading spaces)
        m = RE_DIRECTIVE.match(line)
        if m:
            name = m.group(1).strip().lower()
            rest = m.group(2)
            base_indent = len(line) - len(line.lstrip(" "))

            if name in ("code-block", "code", "sourcecode", "ipython", "pycon", "console", "parsed-literal"):
                # pick language
                lang = (rest or "").strip() or None
                if name == "parsed-literal":
                    lang_disp = "text"
                else:
                    lang_disp = (lang or "").strip() or None
                # skip options -> body
                body_start = skip_directive_options(lines, i + 1, base_indent)
                block, j = dedent_block(lines, body_start, base_indent)
                if block:
                    code_blocks.append({"lang": lang_disp, "code": block})
                    # IMPORTANT: do NOT rstrip, always emit full fence
                    open_fence = "```" + (lang_disp or "")
                    out_text_lines.append(open_fence)
                    out_text_lines.append(block)
                    out_text_lines.append("```")
                i = j
                continue

            if name in strip_directives:
                j = skip_directive_options(lines, i + 1, base_indent)
                _, j2 = dedent_block(lines, j, base_indent)
                i = j2
                continue

            # Other directives: keep body, drop marker+options
            j = skip_directive_options(lines, i + 1, base_indent)
            body, j2 = dedent_block(lines, j, base_indent)
            if body:
                out_text_lines.append(body)
            i = j2
            continue

        # B. Literal block by '::' (not a directive)
        if line.rstrip().endswith("::") and not line.lstrip().startswith(".. "):
            para = re.sub(r"::\s*$", ":", line.rstrip())
            out_text_lines.append(para)
            base_indent = len(line) - len(line.lstrip(" "))
            block, j = dedent_block(lines, i + 1, base_indent)
            if block:
                code_blocks.append({"lang": "text", "code": block})
                out_text_lines.append("```")
                out_text_lines.append(block)
                out_text_lines.append("```")
                i = j
                continue
            i += 1
            continue

        # default
        out_text_lines.append(line)
        i += 1

    # Join lines
    txt = "\n".join(out_text_lines)
    # Clean only outside fences
    txt_clean = _normalize_outside_code(txt)
    return txt_clean, code_blocks

def should_exclude_url(url_html: str, substrings: List[str]) -> bool:
    """Return True if url_html contains any of the substrings (case-insensitive)."""
    if not url_html or not substrings:
        return False
    url_l = url_html.lower()
    return any(s.lower() in url_l for s in substrings if s)

def process_file(
    inp: Path,
    outp: Path,
    strip_names: List[str],
    min_words: int,
    save_code_blocks: bool,
    exclude_url_contains: List[str],
):
    strip_set = {n.strip().lower() for n in strip_names if n.strip()}
    total, kept, skipped_filter, skipped_short = 0, 0, 0, 0
    output_items: List[Dict] = []

    with inp.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue

            # URL filtering
            url_html = j.get("url_html", "") or ""
            if should_exclude_url(url_html, exclude_url_contains):
                skipped_filter += 1
                continue

            rst = j.get("rst_text", "") or ""
            text_clean, code_blocks = parse_rst_keep_code(rst, strip_set or DEFAULT_STRIP_DIRECTIVES)

            # Optional: drop too-short items (unless they contain code blocks)
            if min_words > 0:
                wc = len(text_clean.split())
                if wc < min_words and not code_blocks:
                    skipped_short += 1
                    continue

            # Build slim output item
            section_title = j.get("section_title", "") or ""
            item = {
                "file": url_html.replace("https://lightning.ai/", ""),
                "title": sanitize_title(section_title),
                "text": text_clean,
            }
            if save_code_blocks:
                item["code_blocks"] = code_blocks

            output_items.append(item)
            kept += 1

    # Add index field
    for idx, obj in enumerate(output_items):
        obj["index"] = idx
        obj["label"] = "docs"

    # Write a single JSON array to output
    with outp.open("w", encoding="utf-8") as fout:
        json.dump(output_items, fout, ensure_ascii=False, indent=2)

    print(
        f"Processed {total} items -> kept {kept}, "
        f"filtered_by_url {skipped_filter}, filtered_by_min_words {skipped_short} -> {outp}"
    )

def main():
    ap = argparse.ArgumentParser(description="Clean JSONL with RST text and output a single JSON array.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path (with rst_text).")
    ap.add_argument("--out", dest="out", required=True, help="Output JSON path (JSON array).")
    ap.add_argument("--strip", dest="strip", type=str,
                    default=",".join(sorted(DEFAULT_STRIP_DIRECTIVES)),
                    help="Comma-separated directive names to strip entirely.")
    ap.add_argument("--min-words", dest="min_words", type=int, default=10,
                    help="Drop items with <min_words (and no code blocks). 0=disable.")
    ap.add_argument("--save-code-blocks", dest="save_code_blocks", action="store_true",
                    help="Include code_blocks in output entries when set.")
    ap.add_argument("--exclude-url-contains", dest="exclude_url_contains", type=str, default="/accelerators/,/advanced/",
                    help="Comma-separated substrings; if any is found in url_html (case-insensitive), the item is dropped.")
    args = ap.parse_args()

    strip_names = [x.strip() for x in args.strip.split(",") if x.strip()]
    exclude_url_contains = [x.strip() for x in args.exclude_url_contains.split(",") if x.strip()]

    process_file(
        Path(args.inp),
        Path(args.out),
        strip_names,
        args.min_words,
        args.save_code_blocks,
        exclude_url_contains,
    )

if __name__ == "__main__":
    main()
