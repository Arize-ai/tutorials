"""Unit tests for the factory pieces that don't need Codex, GitHub or Arize AX."""

import shutil
from pathlib import Path

from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CommandExecutionThreadItem,
    FileChangeThreadItem,
)

from factory import agent, backlog, config, setup, tracing
from factory.stations import release, review, router, spec


def test_template_copies_the_app_without_local_state(tmp_path):
    (setup.TEMPLATE / "helpdesk.db").touch()
    try:
        shutil.copytree(setup.TEMPLATE, tmp_path / "repo", ignore=setup.IGNORE)
    finally:
        (setup.TEMPLATE / "helpdesk.db").unlink()
    repo = tmp_path / "repo"
    for needed in (".github/workflows/ci.yml", "AGENTS.md", "pyproject.toml", "uv.lock", "helpdesk/app.py"):
        assert (repo / needed).exists(), needed
    assert not (repo / "helpdesk.db").exists()
    assert not (repo / ".venv").exists()
    # The router's sensitive paths must name files that exist in the app.
    for path in config.DEFAULT_SENSITIVE_PATHS:
        assert (repo / path).exists(), path


def test_strict_schema_requires_everything_and_makes_optional_nullable():
    s = agent.strict_schema(spec.SCHEMA)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(spec.SCHEMA["properties"])
    assert s["properties"]["questions"]["type"] == ["array", "null"]
    assert s["properties"]["risk"]["type"] == "string"  # was required: stays non-null
    assert s["properties"]["risk"]["enum"] == ["low", "medium", "high"]


def test_strict_schema_recurses_into_array_items():
    items = agent.strict_schema(review.SCHEMA)["properties"]["findings"]["items"]
    assert items["additionalProperties"] is False
    assert set(items["required"]) == {"severity", "file", "line", "title", "body"}
    assert items["properties"]["line"]["type"] == ["integer", "null"]


def test_strict_schema_does_not_mutate_the_station_schema():
    before = repr(release.SCHEMA)
    agent.strict_schema(release.SCHEMA)
    assert repr(release.SCHEMA) == before


def test_drop_nulls_removes_nulled_optional_fields():
    out = agent.drop_nulls({"a": 1, "b": None, "c": [{"x": None, "y": 2}]})
    assert out == {"a": 1, "c": [{"y": 2}]}


def test_agent_env_sends_traces_to_the_factory_project_with_safe_attributes(monkeypatch):
    monkeypatch.setattr(tracing.config, "ARIZE_PROJECT_NAME", "software-factory")
    env = tracing.agent_env(
        {
            "issue": 7,
            "station": "build",
            "repo": "me/helpdesk",
            "note": "a, b=c",
            "vid": "UHJvbXB0==",
            "empty": "",
            "none": None,
        }
    )
    assert env["ARIZE_PROJECT_NAME"] == "software-factory"
    pairs = dict(p.split("=", 1) for p in env["OTEL_RESOURCE_ATTRIBUTES"].split(","))
    assert pairs == {
        "factory.issue": "7",
        "factory.station": "build",
        "factory.repo": "me/helpdesk",
        "factory.note": "a_ b=c",
        "factory.vid": "UHJvbXB0==",
    }


def test_describe_item_labels_tool_calls_only(tmp_path):
    cmd = CommandExecutionThreadItem.model_validate(
        {
            "id": "1",
            "type": "commandExecution",
            "command": "uv   run pytest -q",
            "cwd": str(tmp_path),
            "status": "completed",
            "commandActions": [],
        }
    )
    assert agent.describe_item(cmd, tmp_path) == "$ uv run pytest -q"
    change = FileChangeThreadItem.model_validate(
        {
            "id": "2",
            "type": "fileChange",
            "status": "completed",
            "changes": [
                {"path": str(tmp_path / "helpdesk" / "services.py"), "kind": {"type": "update"}, "diff": ""}
            ],
        }
    )
    assert agent.describe_item(change, tmp_path) == "edit " + str(Path("helpdesk") / "services.py")
    msg = AgentMessageThreadItem.model_validate({"id": "3", "type": "agentMessage", "text": "hi"})
    assert agent.describe_item(msg, tmp_path) == ""


def _evidence(**over):
    ev = {
        "spec_risk": "low",
        "files_changed": ["helpdesk/services.py"],
        "diff_lines": 20,
        "tests_delta": 1,
        "ci_runs": 1,
        "ci_fixes": 0,
        "reviews": {k: {"verdict": "approve", "confidence": 0.9} for k in review.KINDS},
    }
    ev.update(over)
    return ev


def test_router_rules_floor():
    assert router.rules(_evidence()) == []
    assert router.rules(_evidence(spec_risk="high")) == ["spec marked high risk"]
    assert router.rules(_evidence(tests_delta=-2)) == ["2 test(s) removed"]
    hits = router.rules(_evidence(files_changed=["pyproject.toml"]))
    assert hits == ["sensitive files: pyproject.toml"]
    flagged = _evidence(reviews={"security": {"verdict": "request_changes", "confidence": 0.5}})
    assert router.rules(flagged) == [
        "reviewer requested changes: security",
        "low reviewer confidence: security",
    ]


def test_next_version():
    assert release.next_version(None, "minor") == "v0.1.0"
    assert release.next_version("v0.3.4", "patch") == "v0.3.5"
    assert release.next_version("v0.3.4", "minor") == "v0.4.0"
    assert release.next_version("0.3.4", "major") == "v1.0.0"


def test_router_rules_flag_auth_and_model_changes():
    hits = router.rules(_evidence(files_changed=["helpdesk/deps.py", "helpdesk/models.py"]))
    assert hits == ["sensitive files: helpdesk/deps.py, helpdesk/models.py"]


def test_backlog_parses_and_routes_are_known():
    items = backlog.items()
    assert len(items) == 10
    assert len({it.key for it in items}) == 10
    for it in items:
        assert it.expect in {"automerge", "human"}, it.key
        assert it.title and it.why and it.body, it.key
        assert "expect:" not in it.body and "why:" not in it.body, it.key


def test_sandbox_overrides(monkeypatch):
    monkeypatch.setattr(agent, "uv_cache_dir", lambda: "/home/me/.cache/uv")
    assert agent.sandbox_overrides("read_only", network=True) == ()
    assert agent.sandbox_overrides("workspace_write", network=False) == (
        'sandbox_workspace_write.writable_roots=["/home/me/.cache/uv"]',
    )
    assert agent.sandbox_overrides("workspace_write", network=True)[0] == (
        "sandbox_workspace_write.network_access=true"
    )
