"""
Hand-built retrieval eval set for ai-report.pdf (U.S. Dept. of Education,
"Artificial Intelligence and the Future of Teaching and Learning," May 2023).

`gold_pages` are 0-indexed page numbers as assigned by langchain's
PyPDFLoader (i.e. the same "page" value shown in the app's "Retrieved
context" panel). Each question was written by locating a specific,
verifiable fact on a specific page -- not "somewhere in the document" --
so hit/miss judgments are unambiguous. Every gold_pages/gold_keywords
value below has been directly verified against the extracted PDF text
(see conversation history) -- an earlier version had 3 page numbers
that were off by an offset-of-3 bug (doc's printed page number vs.
PyPDFLoader's 0-indexed page), now fixed and re-verified.

`gold_keywords` supports a *strict* (content-level) recall check in
eval_retrieval.py, separate from the *loose* (page-level) check.
Page-level recall counts a hit whenever any retrieved chunk merely
comes from the right page, even if the specific chunk happens to be a
neighboring slice that doesn't contain the actual answer-bearing
sentence (a real page can span several chunks). Content-level recall
requires the retrieved chunk's own text to contain a verified
substring from the source, which is what a resume claim about
"retrieval quality" should actually be measured against. Report both;
don't quote only the more flattering one.

Questions deliberately span every major section (front matter,
Foundations, What is AI?, Learning, Teaching, Formative Assessment/AES,
R&D, Recommendations) so a chunking strategy that only helps one part
of the document can't look artificially good.
"""

EVAL_SET = [
    {
        "id": 1,
        "question": "What is this report titled and when was it published?",
        "gold_answer": "Artificial Intelligence and the Future of Teaching and Learning; May 2023",
        "gold_pages": [0],
        "gold_keywords": ["Insights and Recommendations"],
    },
    {
        "id": 2,
        "question": "Who was named as Secretary of Education in this report?",
        "gold_answer": "Miguel A. Cardona",
        "gold_pages": [1],
        "gold_keywords": ["Cardona"],
    },
    {
        "id": 3,
        "question": "How does the report preliminarily define AI?",
        "gold_answer": "automation based on associations",
        "gold_pages": [4, 14],
        "gold_keywords": ["automation based on associations"],
    },
    {
        "id": 4,
        "question": "What 1968 film does the report cite, and what was the AI character called?",
        "gold_answer": "2001: A Space Odyssey; the character was called HAL",
        "gold_pages": [15],
        "gold_keywords": ["Heuristically-programmed"],
    },
    {
        "id": 5,
        "question": "What is the report's key recommendation right after the 'What is AI?' section?",
        "gold_answer": "Human in the Loop AI",
        "gold_pages": [19],
        "gold_keywords": ["Human in the Loop AI"],
    },
    {
        "id": 6,
        "question": "What two books does the report recommend for a synthesis of research on how people learn?",
        "gold_answer": "How People Learn and How People Learn II",
        "gold_pages": [21],
        "gold_keywords": ["How People Learn"],
    },
    {
        "id": 7,
        "question": "What is the key recommendation in the Learning section?",
        "gold_answer": "Seek AI Models Aligned to a Vision for Learning",
        "gold_pages": [27],
        "gold_keywords": ["Seek AI Models Aligned"],
    },
    {
        "id": 8,
        "question": "What specific reduction in weekly teacher preparation hours does the report cite as a potential AI benefit?",
        "gold_answer": "reducing the average 11 hours of weekly preparation down to only six",
        "gold_pages": [31],
        "gold_keywords": ["11 hours"],
    },
    {
        "id": 9,
        "question": "What year was the first vision for AI-based essay-scoring programs, and how many years of effort followed?",
        "gold_answer": "1966; 56 years",
        "gold_pages": [43],
        "gold_keywords": ["56 years"],
    },
    {
        "id": 10,
        "question": "What do Gardner, O'Leary, and Yuan say about AES systems matching human judges?",
        "gold_answer": "arguably remains a long way off",
        "gold_pages": [44],
        "gold_keywords": ["long way off"],
    },
    {
        "id": 11,
        "question": "Name one of the 'Related Questions' the report raises about formative assessment.",
        "gold_answer": "e.g. 'Is formative assessment bringing benefits to the student learning experience...?'",
        "gold_pages": [46],
        "gold_keywords": ["Is formative assessment bringing benefits"],
    },
    {
        "id": 12,
        "question": "What 2010 NETP concept motivates the Research and Development section?",
        "gold_answer": "grand challenges",
        "gold_pages": [47],
        "gold_keywords": ["grand challenges"],
    },
    {
        "id": 13,
        "question": "How many desirable characteristics of AI models for education does Figure 14 highlight?",
        "gold_answer": "six",
        "gold_pages": [58],
        "gold_keywords": ["six desirable characteristics"],
    },
    {
        "id": 14,
        "question": "Name one of the six desirable characteristics of AI models listed in the Recommendations section.",
        "gold_answer": "e.g. 'Safe and Effective Systems' or 'Human Alternatives, Consideration and Feedback'",
        "gold_pages": [59],
        "gold_keywords": ["Human Alternatives, Consideration and Feedback"],
    },
    {
        "id": 15,
        "question": "What ESEA framework does the report reference for evidence standards?",
        "gold_answer": "four tiers of evidence",
        "gold_pages": [12],
        "gold_keywords": ["four tiers of evidence"],
    },
    {
        "id": 16,
        "question": "What Blueprint for an AI Bill of Rights concept does the report say 'humans in the loop' builds on?",
        "gold_answer": "Human Alternatives, Consideration, and Fallback",
        "gold_pages": [10],
        "gold_keywords": ["Human Alternatives, Consideration, and Fallback"],
    },
]
