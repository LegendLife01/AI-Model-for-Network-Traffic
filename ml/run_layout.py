"""Helpers for consistent run-folder layout."""

from __future__ import annotations

from pathlib import Path


SUBDIRS = {
    "raw_data": "raw_data",
    "results": "results",
    "json": "json",
    "images": "images",
    "model": "model",
}


def ensure_run_layout(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {"root": root}
    for key, dirname in SUBDIRS.items():
        path = root / dirname
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def artifact_path(run_dir: str | Path, filename: str, kind: str) -> Path:
    paths = ensure_run_layout(run_dir)
    return paths[kind] / filename


def find_artifact(run_dir: str | Path, filename: str, kind: str | None = None) -> Path:
    root = Path(run_dir)
    candidates: list[Path] = []
    if kind:
        candidates.append(root / SUBDIRS[kind] / filename)
    candidates.append(root / filename)
    for dirname in SUBDIRS.values():
        candidates.append(root / dirname / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
