"""Your code evaluators for the UTM link skill (Lesson 3).

These run on your machine when `run_experiment.py` passes them to `experiments.run`, and
their scores are logged to the Arize AX experiment as one column per evaluator.

`CampaignCheck` is done for you. Write the other three, following the same pattern, and add
them to EVALUATORS at the bottom:

- SourceMediumCheck: utm_source and utm_medium match expected_source and expected_medium.
- ContentCheck: when expected_content is set, utm_content contains it, and the name isn't in
  utm_source or utm_campaign instead. When expected_links is more than 1, every link has a
  different utm_content.
- UrlIntactCheck: one "?", no repeated parameters, every keep_params value survives, the
  fragment survives, and no UTM value has capital letters.

Each dataset row has these columns:

    request_id, request
    expected_source, expected_medium, expected_campaign
    expected_content   a name or placement that must appear in utm_content (or empty)
    expected_links     how many links the request asks for (usually 1)
    keep_params        existing query parameters that must survive, like "plan=team"
    fragment           the #anchor that must survive (or empty)
    should_decline     true when the right answer is not to tag a link

A finished version is in solutions/evaluators.py.
"""

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
            # Code can't tell an example link in an explanation from a link the agent wants
            # you to use, so the LLM judge grades these rows.
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


# TODO: write SourceMediumCheck, ContentCheck, and UrlIntactCheck.


EVALUATORS = [CampaignCheck()]
