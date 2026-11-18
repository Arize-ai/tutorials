"""Station system prompts, served from Arize Prompt Hub.

Each agent station has one Hub prompt (a single SYSTEM message):

    <prefix>-spec  <prefix>-build  <prefix>-babysit  <prefix>-router  <prefix>-release
    <prefix>-review-correctness  <prefix>-review-security  <prefix>-review-tests

where <prefix> is FACTORY_PROMPT_PREFIX (default software-factory).

At run time a station asks for the version carrying the `production` label, so
promoting a better prompt in Arize changes the factory's behaviour with no code change
or restart (lookups are cached for PROMPT_TTL_S). The files in factory/prompts/ are the
seed and the fallback: if Hub can't be reached, the station runs on the local text and
the span says so (prompt_source=local).

Every station span records which prompt ran (prompt_name, prompt_version_id,
prompt_commit, prompt_source), and the station passes the same fields to its agent's
Codex trace, so evals and experiments can compare versions cleanly.

    python -m factory.prompts status            # what production points at, drift vs local
    python -m factory.prompts push              # create missing prompts / new versions from local files
    python -m factory.prompts push --promote    # ...and move `production` to them
    python -m factory.prompts pull              # write production text back to factory/prompts/hub/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from opentelemetry import trace

from . import axcli, config

log = logging.getLogger(__name__)

LABEL = "production"
PROMPT_TTL_S = 60

# station key -> (Hub name, local files concatenated in order, model it targets)
CATALOG: dict[str, tuple[str, list[str], str]] = {
    "spec": (f"{config.PROMPT_PREFIX}-spec", ["spec"], "build"),
    "build": (f"{config.PROMPT_PREFIX}-build", ["build"], "build"),
    "babysit": (f"{config.PROMPT_PREFIX}-babysit", ["babysit"], "build"),
    "router": (f"{config.PROMPT_PREFIX}-router", ["router"], "build"),
    "release": (f"{config.PROMPT_PREFIX}-release", ["release"], "build"),
    "review_correctness": (
        f"{config.PROMPT_PREFIX}-review-correctness",
        ["review_common", "review_correctness"],
        "review",
    ),
    "review_security": (
        f"{config.PROMPT_PREFIX}-review-security",
        ["review_common", "review_security"],
        "review",
    ),
    "review_tests": (f"{config.PROMPT_PREFIX}-review-tests", ["review_common", "review_tests"], "review"),
}


@dataclass(frozen=True)
class Prompt:
    key: str
    name: str
    text: str
    version_id: str = ""
    commit: str = ""
    source: str = "local"  # hub | local

    @property
    def meta(self) -> dict:
        return {
            "prompt_name": self.name,
            "prompt_version_id": self.version_id or None,
            "prompt_commit": self.commit or None,
            "prompt_source": self.source,
            "prompt_sha": sha(self.text),
        }


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def local_text(key: str) -> str:
    _, files, _ = CATALOG[key]
    return "\n".join(config.prompt(f) for f in files)


def model_for(key: str) -> str:
    return config.REVIEW_MODEL if CATALOG[key][2] == "review" else config.BUILD_MODEL


# ---------- runtime ----------

_cache: dict[str, tuple[float, Prompt]] = {}
_lock = threading.Lock()


def _from_hub(key: str) -> Prompt:
    name = CATALOG[key][0]
    v = axcli.run_json("prompts", "get-version-by-label", name, "--label", LABEL)
    v = v.get("version", v)
    system = [m.get("content") or "" for m in v.get("messages", []) if m.get("role") == "SYSTEM"]
    if not system or not system[0].strip():
        raise axcli.AxError(f"{name}@{LABEL} has no SYSTEM message")
    return Prompt(key, name, system[0], v.get("id", ""), (v.get("commit_hash") or "")[:12], "hub")


def get(key: str) -> Prompt:
    """The production prompt for a station, from Hub (cached) or the local fallback."""
    with _lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < PROMPT_TTL_S:
            return hit[1]
    p = None
    if axcli.available():
        try:
            p = _from_hub(key)
        except Exception as e:  # noqa: BLE001 - any Hub failure falls back to the repo copy
            log.warning("Prompt Hub lookup for %s failed, using local text: %s", key, e)
    if p is None:
        p = Prompt(key, CATALOG[key][0], local_text(key))
    with _lock:
        _cache[key] = (time.monotonic(), p)
    return p


def use(key: str) -> Prompt:
    """get(), and stamp the prompt identity on the current (station) span."""
    p = get(key)
    span = trace.get_current_span()
    for k, v in p.meta.items():
        if v is not None:
            span.set_attribute(f"factory.{k}", v)
    return p


# ---------- CLI ----------


def _hub_prompt(name: str) -> dict | None:
    found = axcli.run_json("prompts", "list", "--name", name)
    for p in found if isinstance(found, list) else found.get("prompts", []):
        if p.get("name") == name:
            return p
    return None


def _latest(name: str) -> dict:
    return axcli.run_json("prompts", "get", name).get("version") or {}


def _production(name: str) -> dict | None:
    try:
        v = axcli.run_json("prompts", "get-version-by-label", name, "--label", LABEL)
        return v.get("version", v)
    except axcli.AxError:
        return None


def _system(v: dict | None) -> str:
    return next(
        (m.get("content") or "" for m in (v or {}).get("messages", []) if m.get("role") == "SYSTEM"), ""
    )


def push(promote: bool, message: str) -> None:
    for key, (name, files, _) in CATALOG.items():
        text = local_text(key)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"role": "SYSTEM", "content": text}], f)
        common = [
            "--provider",
            "OPEN_AI",
            "--model",
            model_for(key),
            "--input-variable-format",
            "NONE",
            "--messages",
            f.name,
        ]
        if _hub_prompt(name) is None:
            desc = (
                f"Software factory: system prompt for the {key.replace('_', ' ')} station "
                f"(from {', '.join(x + '.md' for x in files)})"
            )
            v = axcli.run_json(
                "prompts",
                "create",
                "--name",
                name,
                *common,
                "--description",
                desc,
                "--commit-message",
                message or "Initial version from repo",
            ).get("version", {})
            print(f"created  {name:28} {v.get('id')}")
        else:
            latest = _latest(name)
            if _system(latest) == text:
                v = latest
                print(f"same     {name:28} {v.get('id')}")
            else:
                v = axcli.run_json(
                    "prompts",
                    "create-version",
                    name,
                    *common,
                    "--commit-message",
                    message or "Sync from repo",
                )
                v = v.get("version", v)
                print(f"version  {name:28} {v.get('id')}")
        Path(f.name).unlink(missing_ok=True)
        prod = _production(name)
        if (promote or prod is None) and (not prod or prod.get("id") != v.get("id")):
            axcli.run("prompts", "set-version-labels", v["id"], "--label", LABEL, space=False)
            print(f"         {'':28} -> {LABEL}")


def status() -> None:
    for key, (name, _, _) in CATALOG.items():
        prod = _production(name)
        if prod is None:
            print(f"{name:28} (no {LABEL} version)")
            continue
        drift = "in sync with repo" if _system(prod) == local_text(key) else "differs from repo"
        print(f"{name:28} {prod.get('id')}  {(prod.get('commit_message') or '')[:40]!r:44} {drift}")


def pull() -> None:
    out = config.PROMPTS / "hub"
    out.mkdir(exist_ok=True)
    for key, (name, _, _) in CATALOG.items():
        text = _system(_production(name))
        if text:
            (out / f"{key}.md").write_text(text, encoding="utf-8")
            print(f"wrote {out / f'{key}.md'}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("push")
    s.add_argument("--promote", action="store_true", help=f"move `{LABEL}` to the pushed versions")
    s.add_argument("-m", "--message", default="", help="commit message for new versions")
    sub.add_parser("status")
    sub.add_parser("pull")
    args = p.parse_args(argv)
    if not axcli.available():
        p.error("needs the ax CLI and ARIZE_SPACE_ID")
    {"push": lambda: push(args.promote, args.message), "status": status, "pull": pull}[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
