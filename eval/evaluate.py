"""Run the 15 test questions against the /ask endpoint and report the results.

Start the server first (uvicorn main:app), then run from the project folder:
    python eval/evaluate.py
"""
import json
from pathlib import Path

import requests

API_URL = "http://localhost:8000/ask"
EVAL_DIR = Path(__file__).parent


def normalise(text):
    # Lower-case and drop thousands separators, so "NGN 400,000" matches "400000"
    return text.lower().replace(",", "")


def grade(case, response):
    """Return (status, reason) for one question."""
    answered = response["answered"]

    # Questions the policies do NOT cover: the only correct result is a refusal
    if not case["should_answer"]:
        if answered:
            return "FAIL: answered", "Should have refused, but gave an answer"
        return "correct refusal", ""

    # Questions the policies DO cover
    if not answered:
        return "FAIL: refused", "Refused a question the policies answer"

    cited_ids = {c.split(" | ")[0] for c in response["citations"]}
    missing_sources = [s for s in case["expected_sources"] if s not in cited_ids]

    answer = normalise(response["answer"])
    missing_facts = [
        group[0] for group in case["must_include"]
        if not any(normalise(option) in answer for option in group)
    ]

    problems = []
    if missing_sources:
        problems.append("missing citation: " + ", ".join(missing_sources))
    if missing_facts:
        problems.append("missing fact: " + ", ".join(missing_facts))

    if problems:
        return "FAIL: incomplete", "; ".join(problems)
    return "correct answer", ""


def main():
    cases = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    results = []

    for case in cases:
        try:
            r = requests.post(API_URL, json={"question": case["question"]}, timeout=60)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise SystemExit("Could not reach the API. Start it first with: uvicorn main:app")
        response = r.json()
        status, reason = grade(case, response)
        results.append({**case, "response": response, "status": status, "reason": reason})
        print(f"{case['id']:>2}. {status:<17} {case['question']}")

    # ---------- Summary ----------
    answerable = [x for x in results if x["should_answer"]]
    unanswerable = [x for x in results if not x["should_answer"]]
    n_correct = sum(x["status"] == "correct answer" for x in answerable)
    n_refused = sum(x["status"] == "correct refusal" for x in unanswerable)
    failures = [x for x in results if x["status"].startswith("FAIL")]

    summary = [
        f"Answered correctly: {n_correct} / {len(answerable)}",
        f"Correctly refused:  {n_refused} / {len(unanswerable)}",
        f"Overall:            {n_correct + n_refused} / {len(results)}",
    ]
    print("\n" + "\n".join(summary))
    if failures:
        print("\nFailures:")
        for x in failures:
            print(f"  Q{x['id']}: {x['status']} ({x['reason']})")

    # ---------- Save full results ----------
    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# Evaluation results", "", *[f"- {s}" for s in summary], "",
             "| # | Question | Expected | Result | Notes |", "|---|---|---|---|---|"]
    for x in results:
        expected = "answer" if x["should_answer"] else "refuse"
        lines.append(f"| {x['id']} | {x['question']} | {expected} | {x['status']} | {x['reason']} |")
    lines += ["", "## Full answers", ""]
    for x in results:
        lines += [
            f"**Q{x['id']}. {x['question']}**", "",
            f"- Expected: {x['expected_answer']}",
            f"- Answer: {x['response']['answer']}",
            f"- Citations: {', '.join(x['response']['citations']) or 'none'}",
            f"- Result: {x['status']}" + (f" ({x['reason']})" if x["reason"] else ""), "",
        ]
    (EVAL_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved eval/results.json and eval/results.md")


if __name__ == "__main__":
    main()
