"""THROWAWAY: proves CI lists a test that needed a rerun. Removed before the PR is final."""

_tries = {"n": 0}


def test_deliberately_flaky_for_the_rerun_report():
    _tries["n"] += 1
    assert _tries["n"] > 1, "fails on the first try on purpose"
