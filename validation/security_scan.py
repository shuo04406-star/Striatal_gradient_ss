#!/usr/bin/env python3
"""Scan this release tree for blocked data, credentials, paths, and large files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = (".nii", ".nii.gz", ".npy", ".npz", ".mat", ".h5", ".hdf5", ".pkl", ".pickle")
SKIP_CONTENT = {Path(__file__).resolve(), REPO / "GITHUB_SECURITY_SCAN.md"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".tsv", ".yaml", ".yml", ".txt", ".cff", ".gitignore"}
PATTERNS = {
    "private_absolute_path": re.compile(r"(?:[A-Za-z]:[\\/](?:Users|data|mnt)[\\/]|(?<![A-Za-z0-9_.-])/(?:data|mnt)/[A-Za-z0-9_.-]+/)", re.I),
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "github_token": re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    "generic_secret_assignment": re.compile(r"(?i)(?:password|passwd|secret|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def is_blocked(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in BLOCKED_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--large-bytes", type=int, default=10 * 1024 * 1024)
    args = parser.parse_args()

    findings: list[dict[str, object]] = []
    files = [
        p
        for p in REPO.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix.lower() != ".pyc"
    ]
    for path in files:
        relative = path.relative_to(REPO).as_posix()
        size = path.stat().st_size
        if size > args.large_bytes:
            findings.append({"kind": "large_file", "path": relative, "detail": size})
        if is_blocked(path):
            findings.append({"kind": "blocked_data_extension", "path": relative, "detail": path.suffix})
        if path.resolve() in SKIP_CONTENT or (path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"kind": "unexpected_binary", "path": relative, "detail": size})
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"kind": name, "path": relative, "detail": f"line {line}"})

    payload = {"root": ".", "files_scanned": len(files), "large_threshold_bytes": args.large_bytes, "findings": findings}
    if args.json_out:
        if args.json_out.exists():
            raise FileExistsError(f"Refusing to overwrite {args.json_out}")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
