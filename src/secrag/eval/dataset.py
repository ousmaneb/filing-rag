from dataclasses import dataclass
from pathlib import Path

import yaml

from secrag.config import ROOT

QUESTIONS = ROOT / "eval" / "questions.yaml"
TYPES = {"exact_lookup", "comparison", "multi_hop", "temporal", "qualitative", "unanswerable"}


@dataclass
class Question:
    id: str
    type: str
    question: str
    answer: str | None
    evidence: list[dict]

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load(path: Path = QUESTIONS) -> list[Question]:
    questions = [Question(**q) for q in yaml.safe_load(path.read_text())]
    problems = []
    for q in questions:
        if q.type not in TYPES:
            problems.append(f"{q.id}: unknown type {q.type!r}")
        if q.answerable and not q.answer:
            problems.append(f"{q.id}: no answer")
        if q.answerable and not q.evidence:
            problems.append(f"{q.id}: no evidence quotes")
    if problems:
        raise SystemExit(
            f"{path} is not labelled. Write ground truth by hand from the filings:\n"
            + "\n".join(problems)
        )
    return questions
