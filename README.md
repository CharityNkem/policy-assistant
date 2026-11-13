# Kestrel Nova Ltd Policy Assistant

A small retrieval-augmented (RAG) assistant that answers employee questions using only the ten Kestrel Nova Ltd policy documents. Every answer cites the policy section it came from. If the policies do not contain the answer, the assistant says so instead of guessing.

The assistant is exposed through a REST endpoint, `POST /ask`.

**Result:** 15/15 on the provided test questions (12/12 answered correctly, 3/3 correctly refused) in each of the final four evaluation runs, after manual review. See [Evaluation](#evaluation) for the full history, including the failures.

---

## Project structure

```
policy-assistant/
├── policies/                  # the 10 policy documents (.md)
├── rag.py                     # load, chunk, embed, retrieve, answer
├── main.py                    # FastAPI app with POST /ask
├── eval/
│   ├── questions.json         # the 15 test questions with expected answers
│   ├── evaluate.py            # runs the questions against /ask and grades them
│   ├── results.md             # final evaluation report (readable)
│   └── results.json           # final evaluation output (raw responses and grades)
├── policy_assistant.ipynb     # development notebook (exploration and tuning)
├── WRITEUP.md                 # failure analysis, improvements, limitations
├── requirements.txt
└── .env.example
```

## Setup

Requires Python 3.11 and an OpenAI API key.

```bash
conda create -n policyrag python=3.11 -y
conda activate policyrag
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```
OPENAI_API_KEY=sk-...
```

## Running the API

From the project folder:

```bash
uvicorn main:app --reload
```

On startup the server loads, chunks and embeds the policies (a few seconds). Then:

- **Interactive docs:** open http://localhost:8000/docs, choose `POST /ask` → *Try it out*.
- **curl** (macOS/Linux):

  ```bash
  curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "How many leave days can I carry over to next year?"}'
  ```

- **curl** (Windows):

  ```bash
  curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"How many leave days can I carry over to next year?\"}"
  ```

### Response format

```json
{
  "answered": true,
  "answer": "You can carry over up to 5 unused leave days into the next calendar year. However, you must use those carried-over days by 31 March of the following year, or they will be lost...",
  "citations": ["HR-01 | 5. Carrying over leave"]
}
```

When the policies do not cover a question:

```json
{
  "answered": false,
  "answer": "The policy documents provided do not contain an answer to this question.",
  "citations": []
}
```

## Running the evaluation

With the server running, in a second terminal:

```bash
python eval/evaluate.py
```

This prints a result for each question and the totals, and writes `eval/results.md` (full answers next to the expected answers) and `eval/results.json`.

---

## How it works

```
question → retrieve top 4 sections → Guard 1 (score cutoff) → LLM (structured output) → Guard 2 (citation check) → answer
```

**Stack:** LangChain, OpenAI `text-embedding-3-small` (embeddings), `gpt-4o-mini` at temperature 0 (answers), LangChain `InMemoryVectorStore`, FastAPI and uvicorn.

### Design decisions

**Chunking by section, not by size.** The documents are short (about 12 KB in total) and every policy uses numbered `##` sections, so each section becomes one chunk (48 chunks). This keeps each rule intact, gives precise citations, and gives the model less unrelated text to stitch a guess from. The disclaimer line and the header line are dropped. The policy name and section heading are added to the top of each chunk's text, so a chunk such as "Up to 5 unused days may be carried over…" is still recognisably about annual leave when it is embedded.

**Citation format.** Each chunk is labelled `<policy ID> | <section>`, for example `HR-01 | 5. Carrying over leave`. The policy ID and version are read from each document's header line, and the section from its heading. The model sees these labels next to the retrieved text and must copy them into a `citations` field.

**Structured output.** The model returns a Pydantic object with `answered` (bool), `answer` (str) and `citations` (list), not free text. This lets the code check the answer instead of interpreting prose.

**Two guards around the LLM:**

1. **Retrieval cutoff (`min_score = 0.4`).** If even the best-matching section scores below 0.4, the question is refused without calling the LLM.
2. **Citation check.** Every citation must match a section that was actually retrieved. If the model claims an answer but gives no valid citation, the answer is discarded and replaced with the fallback refusal. This means the system cannot return an answer with an invented or missing source.

**Who writes the refusal.** When the LLM itself judges that the sources do not cover the question, its own wording is returned (it can say *what* is not covered). When a guard overrides the LLM, a fixed fallback message is used, because the LLM's text in that case is not trustworthy, or does not exist.

**Choosing `k` and `min_score`.** Both were set on my own development questions, not on the 15 test questions.

- `k = 4`: questions that span two policies (for example taxi fares: Travel §6 points to the Expense policy) need both sections retrieved, while more chunks add irrelevant text the model may cite.
- `min_score = 0.4`: answerable development questions scored 0.61–0.72 and unanswerable ones (dress code, pension, canteen) scored 0.31–0.36. The cutoff sits low in that gap, so that valid but unusually phrased questions are not refused. Near-miss questions (for example "How many bereavement leave days do I get?") pass the cutoff and are refused by the LLM's own judgment.

---

## Evaluation

### Method

`eval/questions.json` holds the 15 test questions with an answer key written from the policy documents. Questions 1–12 should be answered; questions 13–15 (pension, dogs in the office, stock options) are not covered by any policy and should be refused.

`evaluate.py` grades each response automatically:

- **Refusal questions:** correct only if `answered` is false.
- **Answerable questions:** correct only if the expected policy is cited **and** the answer contains the key facts (simple keyword checks, e.g. "31 march" for Q2).

The keyword check is a first pass. Every run was also reviewed by hand, and the manual judgment is the final grade.

### Results

| Run | Automated | After manual review | Notes |
|---|---|---|---|
| 1 | 14/15 | 15/15 | Q10 marked as failing because of a grader bug (see below) |
| 2 | 14/15 | **14/15** | **Q4 falsely refused** (see below) |
| *Fix* | | | *Prompt now explicitly requires citations* |
| 3 | 15/15 | 15/15 | |
| 4 | 14/15 | 15/15 | Q10 correct but left out the NGN 15,000 figure the keyword check looks for |
| 5 | 15/15 | 15/15 | |
| 6 | 15/15 | 15/15 | |

**Final:** answered correctly 12/12, correctly refused 3/3, in every run after the fix (runs 3–6).

`eval/results.md` and `eval/results.json` contain the final run (run 6). Each run overwrites these files, so runs 1–5 were recorded from the terminal output and the reports at the time; their files were not kept.

The three unanswerable questions were refused by the retrieval cutoff in every run, without calling the LLM. The two questions that need two policies (Q11: hotel limit and approvals; Q12: course repayment and leave payout) were answered correctly with both policies cited in every run.

### Where it failed (summary)

- **Q4 (real failure, run 2):** a correct answer was replaced by a refusal because the model returned no valid citation. The prompt had never required citations; adding that rule fixed it.
- **Q10 (grader issues, runs 1 and 4):** correct answers marked as failing by the keyword check ("declaring" vs "declare"; a missing NGN 15,000 figure). Graded correct on manual review.

The full failure analysis, planned improvements and known limitations are in **[WRITEUP.md](WRITEUP.md)**.
