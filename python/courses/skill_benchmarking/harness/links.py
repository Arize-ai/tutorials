"""Helpers for pulling UTM links out of the agent's response.

Each experiment run's output is the agent's final response, followed by a footer line the
harness adds with the agent's session ID:

    Here's your link: https://brindlewick.example.com/pricing?utm_source=...

    [harness] agent session: 3f2c...

Use `tagged_links(output)` in your evaluators to get every Brindlewick link that has UTM
parameters, already parsed.
"""

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

SESSION_FOOTER = "[harness] agent session:"
_URL = re.compile(r"https?://[^\s`'\"<>()\[\]]+")


@dataclass
class Link:
    url: str
    params: dict[str, str]  # every query parameter, for example {"utm_source": "linkedin"}
    pairs: list[tuple[str, str]]  # the same parameters in order, including duplicates
    fragment: str  # the part after "#", or ""


def add_session_footer(response: str, session_id: str) -> str:
    return f"{response.rstrip()}\n\n{SESSION_FOOTER} {session_id}"


def split_output(output: str) -> tuple[str, str]:
    """Split an experiment output into (agent response, agent session ID)."""
    response, _, session = (output or "").rpartition(SESSION_FOOTER)
    if not response:
        return output or "", ""
    return response.rstrip(), session.strip()


def tagged_links(output: str) -> list[Link]:
    """Every Brindlewick link in the response that has UTM parameters, in order."""
    response, _ = split_output(output)
    links: list[Link] = []
    seen: set[str] = set()
    for url in _URL.findall(response):
        url = url.rstrip(".,;:*")
        if "brindlewick" not in url or "utm_" not in url or url in seen:
            continue
        seen.add(url)
        parts = urlsplit(url)
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        links.append(Link(url=url, params=dict(pairs), pairs=pairs, fragment=parts.fragment))
    return links
