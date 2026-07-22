#!/usr/bin/env python3
"""Rebuild corpus/joyce_all.txt from Project Gutenberg.

Downloads the five public-domain works of James Joyce, strips the Gutenberg
front/back matter per work, keeps ASCII only, and concatenates them in
chronological order — reproducing the corpus used in the paper
(2,500,916 characters, vocabulary 82).

Usage:  python3 scripts/fetch_corpus.py
Output: corpus/joyce_all.txt
"""
import os
import re
import sys
import urllib.request

# (Gutenberg id, title) in chronological order.
WORKS = [
    (2817,  "Chamber Music"),
    (2814,  "Dubliners"),
    (4217,  "A Portrait of the Artist as a Young Man"),
    (55945, "Exiles"),
    (4300,  "Ulysses"),
]

# Gutenberg mirrors an id under a few URL shapes; try them in order.
URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
]

START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n")
END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK")


def fetch(work_id):
    last_err = None
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(id=work_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - report and try next mirror
            last_err = e
    raise RuntimeError(f"could not download id {work_id}: {last_err}")


def strip_boilerplate(raw):
    m = START_RE.search(raw)
    body = raw[m.end():] if m else raw
    e = END_RE.search(body)
    if e:
        body = body[: e.start()]
    body = body.replace("\r", "")
    body = body.encode("ascii", "ignore").decode()  # match loadCorpus()'s ASCII drop
    return body.strip()


def main():
    os.makedirs("corpus", exist_ok=True)
    parts = []
    for work_id, title in WORKS:
        sys.stderr.write(f"fetching {title} (#{work_id}) ... ")
        sys.stderr.flush()
        body = strip_boilerplate(fetch(work_id))
        sys.stderr.write(f"{len(body):>9,} chars\n")
        parts.append(body + "\n")

    full = "\n\n".join(parts)
    if full.startswith("cover\n"):  # Gutenberg HTML cover artifact
        full = full[6:].lstrip("\n")

    out = os.path.join("corpus", "joyce_all.txt")
    with open(out, "w") as f:
        f.write(full)
    sys.stderr.write(f"\nwrote {out}: {len(full):,} chars, "
                     f"vocab {len(set(full))}\n")


if __name__ == "__main__":
    main()
