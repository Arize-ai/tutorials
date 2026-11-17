"""The course backlog: ten helpdesk issues of known difficulty and risk.

    python -m factory.backlog list              # what's in backlog/ and the expected route
    python -m factory.backlog file              # file every issue on your helpdesk repo
    python -m factory.backlog file 01 04        # file only these

Each file in backlog/ has front matter with the issue `title`, the route the course
expects the router to take (`expect: automerge` or `human`) and `why`. Only the title
and the body are filed: the expectations are for you, not for the agents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from . import config, gh

BACKLOG = config.ROOT / "backlog"


@dataclass
class Item:
    key: str
    title: str
    expect: str
    why: str
    body: str


def parse(path: Path) -> Item:
    text = path.read_text()
    _, front, body = text.split("---\n", 2)
    meta = dict(line.split(": ", 1) for line in front.strip().splitlines())
    return Item(path.stem.split("-", 1)[0], meta["title"], meta["expect"], meta["why"], body.strip())


def items() -> list[Item]:
    return [parse(p) for p in sorted(BACKLOG.glob("*.md"))]


def cmd_list(_args: argparse.Namespace) -> None:
    for it in items():
        print(f"{it.key}  {it.expect:<9}  {it.title}")
        print(f"               {it.why}\n")


def cmd_file(args: argparse.Namespace) -> None:
    config.require_target()
    chosen = [it for it in items() if not args.keys or it.key in args.keys]
    existing = {
        i["title"]
        for i in json.loads(gh.gh("issue", "list", "--state", "all", "--limit", "500", "--json", "title"))
    }
    for it in chosen:
        if it.title in existing:
            print(f"{it.key}  already filed: {it.title}")
            continue
        url = gh.gh("issue", "create", "--title", it.title, "--body-file", "-", input=it.body)
        print(f"{it.key}  {url}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m factory.backlog")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the backlog and the expected routes").set_defaults(func=cmd_list)
    file_ = sub.add_parser(
        "file", help="file backlog issues on your helpdesk repo (skips ones already filed)"
    )
    file_.add_argument("keys", nargs="*", help="issue numbers from the file names, e.g. 01 04")
    file_.set_defaults(func=cmd_file)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
