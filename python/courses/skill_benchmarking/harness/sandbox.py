"""Build a clean, isolated place for one agent run.

Each run gets two fresh temporary directories:

- A workspace: the files Claude Code can see. It holds the skill under test in
  `.claude/skills/`, with any files the skill ships with, such as its campaign list. Nothing
  else, so Claude can't find other skills, notes, or this course's answers.
- A config directory (`CLAUDE_CONFIG_DIR`): an empty Claude Code config, so the run doesn't
  inherit your own settings, hooks, plugins, or skills. The only things in it are the
  coding-harness-tracing hooks, which send the session to Arize AX, and permission rules
  that stop Claude Code reading anything in your home folder or this course.

The permission rules matter because Claude Code's Read, Glob, and Grep tools accept absolute
paths. Without them, the agent could read this course's solutions, or your dotfiles, from
inside the sandbox.
"""

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness.config import COURSE_DIR

# Claude Code hook events that coding-harness-tracing listens to.
HOOK_EVENTS = {
    "SessionStart": "arize-hook-session-start",
    "UserPromptSubmit": "arize-hook-user-prompt-submit",
    "PreToolUse": "arize-hook-pre-tool-use",
    "PostToolUse": "arize-hook-post-tool-use",
    "PostToolUseFailure": "arize-hook-post-tool-use-failure",
    "Stop": "arize-hook-stop",
    "StopFailure": "arize-hook-stop-failure",
    "SubagentStart": "arize-hook-subagent-start",
    "SubagentStop": "arize-hook-subagent-stop",
    "Notification": "arize-hook-notification",
    "PermissionRequest": "arize-hook-permission-request",
    "PermissionDenied": "arize-hook-permission-denied",
    "UserPromptExpansion": "arize-hook-user-prompt-expansion",
    "PreCompact": "arize-hook-pre-compact",
    "PostCompact": "arize-hook-post-compact",
    "SessionEnd": "arize-hook-session-end",
}


@dataclass
class Sandbox:
    workspace: Path
    config_dir: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)
        shutil.rmtree(self.config_dir, ignore_errors=True)


def _permission_rules() -> dict:
    """Deny reads of your home folder and this course, wherever they live.

    Deny rules always win over allow rules, and Read rules also apply to Glob and Grep. The
    workspace is in your system's temp folder, so it stays readable.
    """
    blocked = sorted({Path.home().resolve(), COURSE_DIR.resolve()})
    return {"deny": [f"Read(/{path}/**)" for path in blocked]}


def _settings() -> dict:
    """settings.json: the tracing hooks installed in this project's venv, plus the read rules."""
    bin_dir = Path(sys.executable).parent
    missing = [name for name in HOOK_EVENTS.values() if not (bin_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"coding-harness-tracing hooks not found in {bin_dir}. Run `uv sync` in {COURSE_DIR}."
        )
    hooks = {
        event: [{"hooks": [{"type": "command", "command": str(bin_dir / name)}]}]
        for event, name in HOOK_EVENTS.items()
    }
    return {"hooks": hooks, "permissions": _permission_rules()}


def create_sandbox(skill_dir: Path) -> Sandbox:
    skill_dir = Path(skill_dir).resolve()
    if not (skill_dir / "SKILL.md").exists():
        raise SystemExit(f"No SKILL.md in {skill_dir}")
    temp_root = Path(tempfile.gettempdir()).resolve()
    if any(temp_root.is_relative_to(p.resolve()) for p in (Path.home(), COURSE_DIR)):
        raise SystemExit(
            f"Your temp folder ({temp_root}) is inside your home folder, so the sandbox's read rules "
            "would block the workspace too. Set TMPDIR to a folder outside your home folder."
        )
    workspace = Path(tempfile.mkdtemp(prefix="brindlewick-ws-"))
    config_dir = Path(tempfile.mkdtemp(prefix="brindlewick-cfg-"))
    shutil.copytree(skill_dir, workspace / ".claude" / "skills" / skill_dir.name)
    (config_dir / "settings.json").write_text(json.dumps(_settings(), indent=2))
    return Sandbox(workspace=workspace, config_dir=config_dir)
