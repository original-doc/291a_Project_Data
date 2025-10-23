#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crawl Lightning docs (HTML) and fetch the matching RST sources from GitHub.

Mapping rule (typical Sphinx build):
  https://lightning.ai/docs/pytorch/{channel}/<rel>.html
-> https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/refs/heads/master/docs/source-pytorch/<rel>.rst

Fallback (optional with --try-sources):
  https://lightning.ai/docs/pytorch/{channel}/_sources/<rel>.rst.txt

This collects up to --max-items items that match include prefixes and not in exclude prefixes.

python get_rst_docs.py --channel stable --max-items 1200
"""
import argparse
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from pathlib import Path

HEADERS = {"User-Agent": "docs-rst-crawler/0.1"}

def is_same_site(url, root):
    pu, pr = urlparse(url), urlparse(root)
    return (pu.scheme, pu.netloc) == (pr.scheme, pr.netloc)

def normalize_rel(html_url, channel_root):
    """
    Convert a full HTML URL to the site-relative path without .html.
    e.g., https://lightning.ai/docs/pytorch/stable/starter/introduction.html
       -> ('starter/introduction', 'starter/introduction.html')
    """
    assert html_url.startswith(channel_root)
    rel = html_url[len(channel_root):]
    if rel.endswith(".html"):
        rel_noext = rel[:-5]
    else:
        rel_noext = rel
    return rel_noext.strip("/"), rel.strip("/")

def looks_doc_page(href: str) -> bool:
    return href and href.endswith(".html") and not href.endswith("index.html")

def should_include(rel_noext: str, includes, excludes) -> bool:
    if includes:
        if not any(rel_noext.startswith(p) for p in includes):
            return False
    if excludes:
        if any(rel_noext.startswith(p) for p in excludes):
            return False
    return True

def build_github_raw(rel_noext: str) -> str:
    return ("https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/"
            "refs/heads/master/docs/source-pytorch/" + rel_noext + ".rst")

def build_sources_txt(channel_root: str, rel_noext: str) -> str:
    # Sphinx _sources convention
    return urljoin(channel_root, "_sources/" + rel_noext + ".rst.txt")

def get_html(url, timeout=20):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def get_text_or_none(url, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and r.text and len(r.text.strip()) > 0:
            return r.text
    except requests.RequestException:
        return None
    return None

def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # h1 or first header
    h1 = soup.find(["h1"])
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    # fallback: title tag
    ti = soup.find("title")
    return (ti.get_text(strip=True) if ti else "")

def crawl_and_fetch(channel: str, max_items: int, includes, excludes,
                    try_sources: bool, out_path: Path):
    site_root = f"https://lightning.ai/docs/pytorch/{channel}/"
    seen = set()
    q = [site_root]
    collected = []
    session = requests.Session()

    # discover HTML doc pages
    while q and len(seen) < 50000 and len(collected) < max_items:
        url = q.pop(0)
        try:
            html = session.get(url, headers=HEADERS, timeout=20)
            html.raise_for_status()
        except requests.RequestException:
            continue

        # collect links
        soup = BeautifulSoup(html.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#"):
                continue
            if "#" in href:
                continue
            abs_url = urljoin(url, href)
            if not abs_url.startswith(site_root):
                continue
            if abs_url in seen:
                continue
            if abs_url.endswith(".txt") or abs_url.endswith(".rst"):
                continue
            if "_sources" in abs_url or "_images" in abs_url:
                continue
            # queue into BFS for more discovery (directories/index pages)
            if abs_url.endswith(".html") and abs_url not in seen:
                q.append(abs_url)
            seen.add(abs_url)

            print(abs_url)

            if looks_doc_page(abs_url):
                rel_noext, rel_html = normalize_rel(abs_url, site_root)
                if should_include(rel_noext, includes, excludes):
                    collected.append(abs_url)
                    if len(collected) >= max_items:
                        break

    # fetch mappings to RST
    out_f = out_path.open("w", encoding="utf-8")
    print(collected)
    with open("collected_urls.txt", "w", encoding="utf-8") as cu_f:
        collected = list(sorted(set(collected)))
        for url in collected:
            cu_f.write(url + "\n")
    for html_url in tqdm(collected, desc="Fetch RST"):
        try:
            html_txt = get_text_or_none(html_url)
            title = extract_title(html_txt or "") if html_txt else ""
            rel_noext, rel_html = normalize_rel(html_url, site_root)
            gh_raw = build_github_raw(rel_noext)
            rst_txt = get_text_or_none(gh_raw)

            status = "ok"
            src_used = "github_raw"
            sources_txt_url = None

            # fallback to _sources/*.rst.txt if needed
            if (not rst_txt or len(rst_txt.strip()) < 10) and try_sources:
                sources_txt_url = build_sources_txt(site_root, rel_noext)
                rst_txt = get_text_or_none(sources_txt_url)
                if rst_txt:
                    status = "ok_sources"
                    src_used = "sphinx_sources"
                else:
                    status = "missing"

            item = {
                "channel": channel,
                "url_html": html_url,
                "url_rel_html": rel_html,
                "url_rst": gh_raw,
                "url_sources_txt": sources_txt_url,
                "source_used": src_used,
                "status": status,
                "section_title": title,
                "rst_text": rst_txt or "",
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            # be polite
            time.sleep(0.1)
        except Exception as e:
            out_f.write(json.dumps({
                "channel": channel,
                "url_html": html_url,
                "error": repr(e)
            }, ensure_ascii=False) + "\n")
    out_f.close()

def parse_args():
    ap = argparse.ArgumentParser(description="Crawl Lightning docs and fetch matching RST from GitHub.")
    ap.add_argument("--channel", default="stable", choices=["stable", "LTS"], help="Docs channel.")
    ap.add_argument("--max-items", type=int, default=120, help="Max number of doc pages to collect.")
    ap.add_argument("--include-prefixes", type=str, default=
                    "starter,common,data,trainer,callbacks,strategies,accelerators,precision,loggers,profiler,advanced,utilities,model,"\
                        "visualize,tuning,cli,debug,advanced,extensions",
                    help="Comma-separated prefixes relative to channel root; empty = no restriction.")
    ap.add_argument("--exclude-prefixes", type=str, default=
                    "notebooks,examples,glossary,faq,tutorials,team-management,production,security,_sources,_images",
                    help="Comma-separated prefixes to exclude.")
    ap.add_argument("--try-sources", action="store_true", help="Fallback to Sphinx _sources/*.rst.txt if GitHub raw missing.")
    ap.add_argument("--out", default="docs_rst.jsonl", help="Output JSONL.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    includes = [p.strip().strip("/") for p in args.include_prefixes.split(",") if p.strip()]
    excludes = [p.strip().strip("/") for p in args.exclude_prefixes.split(",") if p.strip()]
    crawl_and_fetch(args.channel, args.max_items, includes, excludes, args.try_sources, Path(args.out))
