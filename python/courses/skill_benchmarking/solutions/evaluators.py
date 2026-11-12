"""Code evaluators for the UTM link skill (Lesson 3 solution).

These run on your machine when you pass them to `experiments.run`, and their scores are
logged to the Arize AX experiment. Each evaluator checks one part of the link, so AX shows
one column per check and you can filter to, for example, every run with the wrong campaign.

Each evaluator compares the first tagged link in the agent's response with the expected
values in the dataset row:

    expected_source, expected_medium, expected_campaign
    expected_content   a name or placement that must appear in utm_content (or empty)
    expected_links     how many links the request asks for (usually 1)
    keep_params        existing query parameters that must survive, like "plan=team"
    fragment           the #anchor that must survive (or empty)
    should_decline     true when the right answer is not to tag a link

Rows where `should_decline` is true are ignored (score None, label not_applicable). Code can't tell an example
link in an explanation from a link the agent wants you to use, so the LLM judge grades those. The run summary lists them as ignored.
"""

from urllib.parse import parse_qsl

from arize.experiments import EvaluationResult, Evaluator

from harness.links import Link, tagged_links


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _result(passed: bool, explanation: str) -> EvaluationResult:
    return EvaluationResult(score=float(passed), label="pass" if passed else "fail", explanation=explanation)


class LinkEvaluator(Evaluator):
    """Shared setup: ignore rows that shouldn't be tagged, and fail runs with no tagged link."""

    def evaluate(self, *, output, dataset_row, **kwargs) -> EvaluationResult:
        if _truthy(dataset_row.get("should_decline")):
            return EvaluationResult(score=None, label="not_applicable",
                                    explanation="This request shouldn't be tagged. The LLM judge grades it.")
        links = tagged_links(output)
        if not links:
            return _result(False, "The response has no tagged link")
        return self.check(links, dataset_row)

    def check(self, links: list[Link], row) -> EvaluationResult:
        raise NotImplementedError


class CampaignCheck(LinkEvaluator):
    name = "campaign"

    def check(self, links, row):
        got = links[0].params.get("utm_campaign", "")
        want = row["expected_campaign"]
        return _result(got == want, f"utm_campaign={got!r}, expected {want!r}")


class SourceMediumCheck(LinkEvaluator):
    name = "source_medium"

    def check(self, links, row):
        p = links[0].params
        got = (p.get("utm_source", ""), p.get("utm_medium", ""))
        want = (row["expected_source"], row["expected_medium"])
        return _result(got == want, f"source and medium were {got}, expected {want}")


class ContentCheck(LinkEvaluator):
    name = "content"

    def check(self, links, row):
        want = (row.get("expected_content") or "").strip()
        expected_links = int(row.get("expected_links") or 1)
        if expected_links > 1:
            contents = [link.params.get("utm_content", "") for link in links[:expected_links]]
            ok = len(links) >= expected_links and all(contents) and len(set(contents)) == expected_links
            return _result(ok, f"Expected {expected_links} links with different utm_content, got {contents}")
        if not want:
            return _result(True, "utm_content isn't required for this request")
        p = links[0].params
        if want in p.get("utm_source", "") or want in p.get("utm_campaign", ""):
            return _result(False, f"{want!r} is in utm_source or utm_campaign instead of utm_content")
        got = p.get("utm_content", "")
        return _result(want in got, f"utm_content={got!r}, expected it to contain {want!r}")


class UrlIntactCheck(LinkEvaluator):
    name = "url_intact"

    def check(self, links, row):
        link = links[0]
        problems = []
        if link.url.count("?") > 1:
            problems.append("more than one '?'")
        keys = [k for k, _ in link.pairs]
        if len(keys) != len(set(keys)):
            problems.append("a parameter appears twice")
        for key, value in parse_qsl(row.get("keep_params") or ""):
            if link.params.get(key) != value:
                problems.append(f"lost {key}={value}")
        fragment = (row.get("fragment") or "").strip()
        if fragment and link.fragment != fragment:
            problems.append(f"lost #{fragment}")
        if any(v != v.lower() for k, v in link.pairs if k.startswith("utm_")):
            problems.append("a UTM value has capital letters")
        return _result(not problems, "; ".join(problems) or "The rest of the URL is intact")


EVALUATORS = [CampaignCheck(), SourceMediumCheck(), ContentCheck(), UrlIntactCheck()]
