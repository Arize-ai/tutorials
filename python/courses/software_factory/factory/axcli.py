"""Thin wrapper over the `ax` CLI (Arize AX), used by the Prompt Hub loader and the
outcome harvester. ax authenticates with its own profile (`ax profiles show`); we only
pass the space from .env.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from . import config


class AxError(RuntimeError):
    pass


def available() -> bool:
    return bool(shutil.which("ax") and config.ARIZE_SPACE_ID)


def run(*args: str, space: bool = True, timeout: int = 60) -> str:
    cmd = ["ax", *args]
    if space:
        cmd += ["--space", config.ARIZE_SPACE_ID]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0 or "✗ Error" in p.stdout:
        raise AxError(f"{' '.join(args[:3])}: {(p.stderr or p.stdout).strip()[:500]}")
    return p.stdout


def run_json(*args: str, space: bool = True, timeout: int = 60) -> Any:
    """Run and decode JSON. Commands that already emit JSON (e.g. `spans export --stdout`)
    don't take -o, so it is only added when --stdout isn't present."""
    extra = () if "--stdout" in args else ("-o", "json")
    out = run(*args, *extra, space=space, timeout=timeout)
    # ax prints an upgrade nag after the JSON; decode just the first value.
    start = min((i for i in (out.find("{"), out.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise AxError(f"no JSON in ax output: {out[:200]}")
    value, _ = json.JSONDecoder().raw_decode(out[start:])
    return value
