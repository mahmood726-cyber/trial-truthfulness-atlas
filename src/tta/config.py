# sentinel:skip-file — canonical path defaults; every default is overridable via TTA_* env vars
"""Path + threshold configuration. All values overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path

AACT_SNAPSHOT_DIR = Path(os.environ.get("TTA_AACT_DIR", r"D:\AACT-storage\AACT\2026-04-12"))
PAIRWISE70_DIR = Path(os.environ.get("TTA_PAIRWISE70_DIR", r"C:\Projects\Pairwise70\data"))
DOSSIERGAP_FIXTURE = Path(
    os.environ.get(
        "TTA_DOSSIERGAP_FIXTURE",
        r"C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv",
    )
)

DATA_DIR = Path(os.environ.get("TTA_DATA_DIR", "data"))
OUTPUTS_DIR = Path(os.environ.get("TTA_OUTPUTS_DIR", "outputs"))
JUDGE_CACHE_DIR = Path(os.environ.get("TTA_JUDGE_CACHE_DIR", "data/judge_cache"))

OLLAMA_URL = os.environ.get("TTA_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("TTA_OLLAMA_MODEL", "gemma2:9b")

N_DRIFT_THRESHOLD = float(os.environ.get("TTA_N_DRIFT_THRESHOLD", "0.10"))
DIRECTION_EPSILON = float(os.environ.get("TTA_DIRECTION_EPSILON", "0.01"))

# Minimum bridge confidence below which Flag 0 results are treated as
# "unbridgeable" rather than usable. v0.1.x DossierGap-direct match always
# returns 0.99, so this threshold is dormant; bridge methods 2-4 in v0.2.0
# will return calibrated lower confidences and this gate becomes load-bearing.
BRIDGE_CONFIDENCE_MIN = float(os.environ.get("TTA_BRIDGE_CONFIDENCE_MIN", "0.7"))

# Preflight refuses an AACT snapshot older than this many days; protects
# against stale data drift on long-lived environments. Wired into
# preflight._check_aact via the snapshot mtime check planned for v0.2.0.
AACT_MAX_AGE_DAYS = int(os.environ.get("TTA_AACT_MAX_AGE_DAYS", "90"))

SEED = int(os.environ.get("TTA_SEED", "42"))

# Snapshot date drives the FDAAA results-posting deadline calculation.
# Default to the AACT snapshot's directory date; override via env or CLI when
# rolling forward to a newer snapshot in v0.2.0.
SNAPSHOT_DATE = os.environ.get("TTA_SNAPSHOT_DATE", "2026-04-12")

# When True, OllamaClient will refuse to call URLs that don't resolve to
# loopback (127.0.0.1, ::1, localhost). Opt-in escape hatch for users who
# genuinely run a remote ollama; default-on prevents accidental exfiltration
# of trial data.
ALLOW_REMOTE_OLLAMA = os.environ.get("TTA_ALLOW_REMOTE_OLLAMA", "0") == "1"
