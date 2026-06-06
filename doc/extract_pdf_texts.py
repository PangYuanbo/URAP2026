from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > 180:
        name = name[:180].rstrip()
    return name


def main() -> int:
    root = Path(".").resolve()
    outdir = root / "_texts"
    outdir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(root.glob("*.pdf"))
    for pdf in pdfs:
        r = PdfReader(str(pdf))
        parts: list[str] = []
        for i, page in enumerate(r.pages, start=1):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            parts.append(f"\n\n===== PAGE {i} / {len(r.pages)} =====\n\n")
            parts.append(t)
        out = outdir / (safe_name(pdf.stem) + ".txt")
        out.write_text("".join(parts), encoding="utf-8", errors="ignore")
    print(f"Wrote {len(pdfs)} text files to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

