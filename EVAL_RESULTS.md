# Retrieval & Groundedness Evaluation

This document reports a small, hand-built evaluation of the RAG pipeline's
retrieval quality and answer groundedness, run against the deployed
corpus (`data/ai-report.pdf`, the U.S. Dept. of Education's May 2023
report *"Artificial Intelligence and the Future of Teaching and
Learning"*). Code lives in [`eval/`](eval/).

## Eval set

16 hand-written questions, each anchored to a specific, verifiable fact
on a specific page of the source PDF, spanning every major section of
the document (front matter, Foundations, "What is AI?", Learning,
Teaching, Formative Assessment / Automated Essay Scoring, R&D,
Recommendations) so no single strategy can look artificially good by
only being tested on one part of the document. Full question list and
gold answers: [`eval/eval_set.py`](eval/eval_set.py).

Every `gold_pages` and `gold_keywords` value was verified directly
against the extracted PDF text before being used to score anything. An
earlier draft of this eval set had 3 questions with page numbers off by
a constant offset (the document's own printed page numbers vs. the
0-indexed page metadata the retriever actually returns) — this was
caught by manually re-extracting and diffing each gold page's text
against the expected fact, and fixed before any numbers below were
generated. If you're skimming this for signal on how the evaluation was
done: that mistake and its fix are as relevant as the final numbers.

## Metric 1: Retrieval recall — two definitions, on purpose

Two chunking configurations were built and compared, retrieving with
`sentence-transformers/all-MiniLM-L6-v2` embeddings over a FAISS index:

- **500/50** — chunk size 500 chars, overlap 50 (the app's default)
- **1000/200** — chunk size 1000 chars, overlap 200

Two different recall metrics were measured, deliberately, because they
disagree by a lot and the gap itself is a finding:

- **Page-level recall**: a hit if any retrieved chunk merely comes from
  (within 1 page of) the correct source page. Generous — a page often
  spans several chunks, so this can count a hit even when the specific
  chunk retrieved doesn't contain the answer-bearing sentence.
- **Content-level recall**: a hit only if the retrieved chunk's actual
  text contains a verified substring from the source (e.g. the exact
  phrase "56 years" for the AES-history question). Stricter, and closer
  to what "the retriever found the answer" should actually mean.

### Results (k = number of chunks retrieved)

| k | 500/50 — page-level | 500/50 — content-level | 1000/200 — page-level | 1000/200 — content-level |
|---|---|---|---|---|
| 2 | 81% (13/16) | 50% (8/16) | 69% (11/16) | 50% (8/16) |
| 4 | 81% (13/16) | **62% (10/16)** | 69% (11/16) | 50% (8/16) |
| 8 | 81% (13/16) | 62% (10/16) | 81% (13/16) | 56% (9/16) |

**The app's default (500/50, k=4) is the best-performing configuration
on the metric that matters — 62% content-level recall vs. 50% for
1000/200.** No config change was made as a result; this confirms the
existing default rather than overturning it.

**The more important finding is the gap itself**: page-level recall
overstates true retrieval accuracy by up to 19 points (81% vs. 62% for
500/50 at k=4). A retrieval eval that only checks "did we land on the
right page" will report a meaningfully better number than a retrieval
eval that checks "did we actually retrieve the sentence that answers
the question." Reporting only the page-level number would have been
misleading.

## Metric 2: Groundedness — LLM-as-judge

For each of the 16 questions: retrieve context (500/50, k=4), generate
an answer with the deployed prompt/model (`openai/gpt-oss-20b` via
Groq), then make a **second, independent** call asking the same model
to judge whether every claim in the answer is supported by the
retrieved context, or whether the answer is a legitimate "I don't know"
given context that genuinely lacks the requested information.

**Result: 15/16 (94%) judged grounded.**

The one failure (Q12) is a judge false-positive on phrasing, not a real
hallucination: the answer correctly cited the 2010 NETP "grand
challenges" concept from the context, but added the connective framing
"motivates the Research and Development section," which the judge
flagged as an unsupported inference. The underlying facts were all
present in the context; the judge penalized reasonable paraphrase.

### Why this number should be reported carefully

This is a **self-consistency check, not verified ground truth**. The
judge is the same underlying model as the generator, just a separate
call — it can share the generator's blind spots, and as Q12 shows, it
can be overly literal about phrasing in ways that don't reflect a real
accuracy problem. Report this as *"15/16 answers judged grounded by an
LLM-as-judge check,"* not as *"15/16 answers are correct"* — those are
different claims, and the second one is not what this measures.

## Honest limitations of this whole evaluation

- **N=16 is small.** A single question flipping hit/miss moves the
  percentage by ~6 points. Treat differences smaller than 2 questions
  (12 points) between configurations as noise, not a real effect.
- **One document, one domain.** These numbers describe retrieval
  quality on this 71-page policy report. They are not a general claim
  about the pipeline's performance on other corpora.
- **The gold set was hand-built by inspecting the source**, not sampled
  from real user queries — it's a controlled test of whether retrieval
  works on known-answerable questions, not a measure of how the system
  handles the messier, more ambiguous questions real users would ask.
- **LLM-as-judge groundedness has no independent ground truth.** See
  above.

## Reproducing this

```bash
python3 build_index.py --chunk-size 500 --chunk-overlap 50 --out faiss_index_500_50
python3 build_index.py --chunk-size 1000 --chunk-overlap 200 --out faiss_index_1000_200
cd eval
python3 eval_retrieval.py 500_50=../faiss_index_500_50 1000_200=../faiss_index_1000_200
python3 eval_groundedness.py ../faiss_index_500_50
```
