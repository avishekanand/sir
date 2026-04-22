"""
Documentation Coverage Checker
================================
Reads docs/MANIFEST.md and reports the current state of documentation
coverage. Exits with code 1 if any ❌ missing entries are found, so it
can be wired into CI.

Usage:
    python scripts/check_docs.py
    python scripts/check_docs.py --fail-on-partial
"""

import re
import sys
import os
import argparse
from typing import List, Tuple

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "MANIFEST.md")

STATUS_DONE    = "✅"
STATUS_PARTIAL = "⚠️"
STATUS_MISSING = "❌"


def parse_manifest(path: str) -> Tuple[List[dict], str]:
    """Parse MANIFEST.md table rows into dicts. Returns (rows, last_updated)."""
    rows = []
    last_updated = "unknown"

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Extract last updated date
    m = re.search(r"_Last updated:\s*(.+?)_", content)
    if m:
        last_updated = m.group(1).strip()

    current_section = "Unknown"
    for line in content.splitlines():
        # Track section headers
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue

        # Skip non-table lines
        if not line.startswith("|") or line.startswith("| ---") or line.startswith("|---"):
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue

        # Skip header rows (first cell is bold or "Component" / "Command" etc.)
        first = cells[0].lower()
        if first in ("component", "command", "dataset", "category", "i want to..."):
            continue
        if first.startswith("**"):
            continue

        # Determine status from last cell
        last_cell = cells[-1]
        if STATUS_DONE in last_cell:
            status = "done"
        elif STATUS_PARTIAL in last_cell:
            status = "partial"
        elif STATUS_MISSING in last_cell:
            status = "missing"
        else:
            continue  # not a status row (e.g. summary table)

        rows.append({
            "section": current_section,
            "name":    cells[0],
            "status":  status,
            "raw":     line,
        })

    return rows, last_updated


def main():
    parser = argparse.ArgumentParser(description="Check documentation coverage from MANIFEST.md")
    parser.add_argument("--fail-on-partial", action="store_true",
                        help="Exit 1 if any partial entries exist (stricter mode)")
    args = parser.parse_args()

    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: MANIFEST not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(2)

    rows, last_updated = parse_manifest(MANIFEST_PATH)

    if not rows:
        print("No status rows found in MANIFEST.md — is the file formatted correctly?")
        sys.exit(2)

    done    = [r for r in rows if r["status"] == "done"]
    partial = [r for r in rows if r["status"] == "partial"]
    missing = [r for r in rows if r["status"] == "missing"]
    total   = len(rows)

    print(f"RAGtune Documentation Coverage  (manifest: {last_updated})")
    print("=" * 56)
    print(f"  ✅ Documented : {len(done):3d} / {total}  ({100*len(done)//total}%)")
    print(f"  ⚠️  Partial    : {len(partial):3d} / {total}  ({100*len(partial)//total}%)")
    print(f"  ❌ Missing    : {len(missing):3d} / {total}  ({100*len(missing)//total}%)")
    print()

    if missing:
        print("Missing documentation:")
        by_section: dict = {}
        for r in missing:
            by_section.setdefault(r["section"], []).append(r["name"])
        for section, names in by_section.items():
            print(f"  [{section}]")
            for name in names:
                print(f"    ❌ {name}")
        print()

    if partial and args.fail_on_partial:
        print("Partial documentation (--fail-on-partial is set):")
        for r in partial:
            print(f"  ⚠️  [{r['section']}] {r['name']}")
        print()

    has_failures = bool(missing) or (args.fail_on_partial and bool(partial))
    if has_failures:
        print("Result: FAIL — update docs/MANIFEST.md when documentation is added.")
        sys.exit(1)
    else:
        print("Result: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
