#!/usr/bin/env python3
"""Build the tab-separated release-file manifest with SHA-256 provenance."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "GITHUB_FILE_MANIFEST.tsv"
CODE_MAP = REPO / "provenance" / "RELEASE_CODE_MAP.tsv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_role(relative: str) -> str:
    if relative.endswith((".py", ".sh", ".R")):
        return "code"
    if relative.startswith("provenance/"):
        return "provenance"
    if relative.startswith("config/") or relative.startswith("environments/") or relative == ".gitignore":
        return "configuration"
    if relative.startswith("GITHUB_"):
        return "release_report"
    return "documentation"


def default_cohort(relative: str) -> str:
    if relative.startswith("src/reference375/"):
        return "375"
    if relative.startswith("src/clinical64/"):
        return "64"
    return "both_or_release"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    with CODE_MAP.open("r", encoding="utf-8", newline="") as handle:
        code_rows = {row["release_path"]: row for row in csv.DictReader(handle, delimiter="\t")}

    rows: list[dict[str, str | int]] = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        if path == OUTPUT:
            continue
        relative = path.relative_to(REPO).as_posix()
        source = code_rows.get(relative, {})
        rows.append(
            {
                "release_path": relative,
                "file_role": file_role(relative),
                "release_classification": source.get("release_classification", "SUPPORTING"),
                "cohort": source.get("cohort", default_cohort(relative)),
                "original_record_id": source.get("original_record_id", "RELEASE_DOCUMENT"),
                "original_sha256": source.get("original_sha256", "NA"),
                "release_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_subject_level_data": "NO",
                "included_in_release": "YES",
            }
        )

    fields = (
        "release_path",
        "file_role",
        "release_classification",
        "cohort",
        "original_record_id",
        "original_sha256",
        "release_sha256",
        "size_bytes",
        "contains_subject_level_data",
        "included_in_release",
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} output={OUTPUT.name} note=manifest_excludes_itself_and_ignored_cache_files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
