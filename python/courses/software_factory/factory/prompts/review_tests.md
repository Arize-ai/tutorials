**Tests.** Does every acceptance criterion have a test that would fail without the
change? Look for tests that were deleted, skipped, xfailed or weakened (looser
asserts, removed cases, broadened mocks), tests that only assert the code runs,
tests that mock away the thing under test, network access in tests, and flaky
time- or order-dependent tests. Compare the test count and assertions against
`origin/main` with `git diff origin/main...HEAD -- tests`.

The other reviewers cover general correctness and security.
