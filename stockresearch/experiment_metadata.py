"""Build provenance-rich, reproducible metadata for each research run."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _git_commit() -> str | None:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=3,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_experiment_metadata(
    data_dir: str | Path,
    *,
    stage: str,
    tickers: list[str],
    test_ids: list[str],
    research_question: str = "",
    hypothesis: str = "",
    notes: str = "",
    started_at: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Capture dataset identity, provenance, protocol and execution environment."""
    root = Path(data_dir)
    panel = root / "panel_adjclose.csv"
    if not panel.is_file():
        raise FileNotFoundError(f"Cannot fingerprint missing price panel: {panel}")

    stat = panel.stat()
    data_metadata = _read_json(root / "metadata.json")
    download_manifest = _read_json(root / "download_manifest.json")
    return {
        "schema_version": "1.0",
        "experiment_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed_seconds,
        "research_protocol": {
            "question": research_question.strip() or None,
            "hypothesis": hypothesis.strip() or None,
            "notes": notes.strip() or None,
            "stage": stage,
            "tickers": list(tickers),
            "method_ids": list(test_ids),
            "random_seed": config.SEED,
            "engine_configuration": {
                "minimum_observations_distribution": 60,
                "minimum_observations_time_series": 120,
                "minimum_observations_risk": 252,
                "chronological_holdout_fraction": 0.30,
                "ewma_lambda": 0.94,
                "risk_var_alpha": 0.05,
                "hmm_random_state": 17,
            },
        },
        "dataset": {
            "path": str(panel),
            "sha256": _sha256(panel),
            "size_bytes": stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "source_metadata": data_metadata,
            "download_manifest": download_manifest,
        },
        "software": {
            "git_commit": _git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "reproducibility_notes": [
            "The dataset fingerprint identifies the exact adjusted-price panel bytes used.",
            "Source metadata is copied when available; vendor history may be revised after retrieval.",
            "A missing Git commit means the run was not executed from a discoverable Git checkout.",
            "Research question and hypothesis are user-supplied and are not inferred from results.",
        ],
    }
