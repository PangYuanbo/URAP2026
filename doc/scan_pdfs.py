from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PaperScan:
    file: str
    pages: int
    text_pages: int
    abstract: str
    first_page_snippet: str


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extract_abstract(text: str) -> str:
    """
    Best-effort extraction across arXiv/CVF/IEEE PDF text layouts.
    """
    t = text or ""
    # Normalise some common ligatures/soft hyphens
    t = t.replace("\u00ad", "")

    # Try a few patterns.
    patterns = [
        # "Abstract ... 1 Introduction"
        r"(?is)\babstract\b[:\s]*(.+?)\b(?:1\s+introduction|i\.\s+introduction|introduction)\b",
        # "ABSTRACT ... Index Terms"
        r"(?is)\babstract\b[:\s]*(.+?)\b(?:index\s+terms|keywords)\b",
        # "Abstract ... 1."
        r"(?is)\babstract\b[:\s]*(.+?)\n\s*1\s+",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return _norm_ws(m.group(1))[:4000]

    # If we can't find an abstract boundary, fall back to the first ~1200 chars.
    return _norm_ws(t)[:1200]


def scan_pdf(path: Path, *, max_pages: int = 3) -> PaperScan:
    r = PdfReader(str(path))
    n_pages = len(r.pages)
    take = min(int(max_pages), n_pages)
    texts: list[str] = []
    for i in range(take):
        try:
            texts.append(r.pages[i].extract_text() or "")
        except Exception:
            texts.append("")

    joined = "\n".join(texts)
    first = _norm_ws(texts[0] if texts else "")[:600]
    abstract = extract_abstract(joined)
    return PaperScan(
        file=path.name,
        pages=n_pages,
        text_pages=take,
        abstract=abstract,
        first_page_snippet=first,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("."))
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("scans.json"))
    args = ap.parse_args()

    root = args.dir.resolve()
    pdfs = sorted(root.glob("*.pdf"))
    scans = [scan_pdf(p, max_pages=args.max_pages) for p in pdfs]

    args.out.write_text(json.dumps([asdict(s) for s in scans], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

