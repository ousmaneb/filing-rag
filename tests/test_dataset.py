import pytest

from secrag.eval.dataset import QUESTIONS, load


def write(tmp_path, text):
    path = tmp_path / "questions.yaml"
    path.write_text(text)
    return path


def test_loader_refuses_answerable_question_without_labels(tmp_path):
    path = write(
        tmp_path,
        """
- {id: q1, type: exact_lookup, question: "AMD revenue?", answer: null, evidence: []}
""",
    )
    with pytest.raises(SystemExit, match="q1: no answer"):
        load(path)


def test_unanswerable_question_needs_no_labels(tmp_path):
    path = write(
        tmp_path,
        """
- id: q1
  type: exact_lookup
  question: AMD FY2024 revenue?
  answer: $25.8 billion
  evidence: [{doc_id: AMD_FY2024, quote: "Net revenue"}]
- {id: q2, type: unanswerable, question: "Apple revenue?", answer: null, evidence: []}
""",
    )
    assert [q.answerable for q in load(path)] == [True, False]


def test_shipped_question_set_is_refused_until_labelled():
    with pytest.raises(SystemExit):
        load(QUESTIONS)
