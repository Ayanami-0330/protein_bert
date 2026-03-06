#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Set


def _extract_repo_ids_from_script(path: Path) -> Set[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    repos: Set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue

        name = node.targets[0].id

        if name == "MODEL_NAME" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            repos.add(node.value.value)
            continue

        if name.endswith("MODEL_CANDIDATES") and isinstance(node.value, ast.List):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    repos.add(elt.value)

    return {repo for repo in repos if "/" in repo}


def _repo_to_cache_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _candidate_hub_roots() -> List[Path]:
    roots: List[Path] = []

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")

    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        tpath = Path(transformers_cache)
        roots.append(tpath if tpath.name == "hub" else tpath / "hub")

    roots.append(Path("/home/nemophila/data/hf_cache/hub"))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    deduped: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _iter_baseline_scripts(lm_dir: Path, script_names: Iterable[str] | None) -> List[Path]:
    if script_names:
        selected = []
        for script_name in script_names:
            path = lm_dir / script_name
            if not path.exists():
                raise FileNotFoundError(f"Script not found: {path}")
            selected.append(path)
        return selected

    return sorted(lm_dir.glob("run_*_baseline.py"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete HuggingFace model caches used by lm_baseline scripts, "
            "while preserving project output files."
        )
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=None,
        help="Optional script names under scripts/lm_baseline (e.g., run_ankh_baseline.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print directories that would be removed.",
    )
    args = parser.parse_args()

    lm_dir = Path(__file__).resolve().parent
    project_dir = lm_dir.parents[1]
    output_dir = project_dir / "cache" / "lm_baseline"

    scripts = _iter_baseline_scripts(lm_dir, args.scripts)
    repos: Set[str] = set()
    for script in scripts:
        repos.update(_extract_repo_ids_from_script(script))

    if not repos:
        print("No HuggingFace repo IDs found in selected scripts.")
        return

    hub_roots = _candidate_hub_roots()
    to_remove: List[Path] = []
    for repo in sorted(repos):
        dirname = _repo_to_cache_dirname(repo)
        for hub_root in hub_roots:
            target = hub_root / dirname
            if target.exists():
                to_remove.append(target)

    print("Selected scripts:")
    for script in scripts:
        print(f"  - {script.name}")

    print("\nDetected model repos:")
    for repo in sorted(repos):
        print(f"  - {repo}")

    print(f"\nPreserving outputs under: {output_dir}")

    if not to_remove:
        print("\nNo matching model cache directories found.")
        return

    print("\nModel cache directories to remove:")
    for path in to_remove:
        print(f"  - {path}")

    if args.dry_run:
        print("\nDry-run mode: no files removed.")
        return

    removed = 0
    for path in to_remove:
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1

    print(f"\nRemoved {removed}/{len(to_remove)} cache directories.")


if __name__ == "__main__":
    main()
