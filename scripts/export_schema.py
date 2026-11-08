"""Write the PostgreSQL DDL of the planner database (generated from service/db.py) to docs/database_schema.sql.

    python scripts/export_schema.py           # write the file
    python scripts/export_schema.py --check   # exit 1 if the file is out of date (used by the tests)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.dialects import postgresql  # noqa: E402
from sqlalchemy.schema import CreateIndex, CreateTable  # noqa: E402

from service.db import Base  # noqa: E402

OUT = ROOT / "docs" / "database_schema.sql"
HEADER = """-- Planner database schema (PostgreSQL). GENERATED from service/db.py by scripts/export_schema.py; do not edit.
-- The service creates these tables itself on start-up (SQLite or PostgreSQL); this file documents them.
-- Transaction boundaries: see service/workflow.py (one transaction per state change, audit events included).
"""


def ddl() -> str:
    dialect = postgresql.dialect()
    parts = [HEADER]
    for table in Base.metadata.sorted_tables:
        parts.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";\n")
        for index in sorted(table.indexes, key=lambda i: i.name):
            parts.append(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";\n")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if docs/database_schema.sql is out of date")
    args = ap.parse_args()
    text = ddl()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; run python scripts/export_schema.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
