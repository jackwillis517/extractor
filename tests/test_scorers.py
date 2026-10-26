import pytest

from scorers import exact, number, overlap, score_posting

JOB = {
    "title": "Software Engineer",
    "company": "Temu",
    "location": "remote",
    "is_remote": True,
    "description": "Full-time remote role for Software Engineer (Ads)",
    "min_salary": 100000,
    "max_salary": 300000,
    "requirements": "2+ years of experience\nJava, C++, Go, or Python",
}


def test_exact_ignores_case_and_spaces():
    assert exact("software  engineer", "Software Engineer") == 1.0
    assert exact("Software Engineer", "Senior Software Engineer") == 0.0
    assert exact(True, True) == 1.0
    assert exact(False, True) == 0.0


@pytest.mark.parametrize(
    "pred, exp, want",
    [
        (100000, 100000, 1.0),
        (104000, 100000, 1.0),  # within 5%
        (110000, 100000, 0.0),
        (0, None, 1.0),  # 0 and None both mean "not listed"
        (None, 100000, 0.0),  # missed a salary
        (90000, 0, 0.0),  # invented a salary
    ],
)
def test_salary(pred, exp, want):
    assert number(0.05)(pred, exp) == want


def test_overlap():
    assert overlap("Python; SQL", "Python\nSQL") == 1.0  # format doesn't matter
    assert overlap("Python", "Python SQL") == pytest.approx(2 / 3)
    assert overlap("C#", "C++") == 0.0
    assert overlap("", None) == 1.0


def test_perfect_posting():
    scores = score_posting({"jobs": [JOB]}, {"jobs": [JOB]})
    assert set(scores.values()) == {1.0}


def test_failed_extraction_scores_zero():
    scores = score_posting(None, {"jobs": [JOB]})
    assert set(scores.values()) == {0.0}


def test_missing_and_extra_jobs():
    missing = score_posting({"jobs": [JOB]}, {"jobs": [JOB, JOB]})
    assert missing["title"] == 0.5 and missing["job_count"] == 0.0
    extra = score_posting({"jobs": [JOB, JOB]}, {"jobs": [JOB]})
    assert extra["title"] == 1.0 and extra["job_count"] == 0.0
