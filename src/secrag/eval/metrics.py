import json

import anthropic

from secrag.config import JUDGE_MODEL
from secrag.eval.dataset import Question
from secrag.generate import Answer, format_excerpts

# The three scores are independent on purpose. An answer that matches the reference but
# isn't supported by the excerpts is correct and unfaithful, and that has to show up.
SYSTEM = """You grade answers from a question-answering system over SEC 10-K filings. Score three things independently; a failure on one must not change the others.

correct: true if the answer agrees with the reference answer on every figure, unit, period and entity the question asks about. Ignore wording, and ignore extra detail that does not contradict the reference.

faithful: true if every factual claim in the answer is supported by the excerpts. Judge this against the excerpts only, not the reference answer and not your own knowledge. An answer can be correct and unfaithful.

citation_valid: true if every [n] marker refers to an excerpt that exists and supports the claim it is attached to, and no factual claim is left without one."""

SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "citation_valid": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["correct", "faithful", "citation_valid", "reasoning"],
    "additionalProperties": False,
}


def judge(question: Question, answer: Answer, client: anthropic.Anthropic) -> dict:
    prompt = (
        f"<question>{question.question}</question>\n"
        f"<reference_answer>{question.answer}</reference_answer>\n"
        f"<excerpts>\n{format_excerpts(answer.hits)}\n</excerpts>\n"
        f"<answer>{answer.text}</answer>"
    )
    response = client.beta.messages.create(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    return json.loads(next(block.text for block in response.content if block.type == "text"))
