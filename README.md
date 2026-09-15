# RAG Chatbot

A retrieval-augmented generation chatbot for question-answering over a
document, built with LangChain, FAISS, and Groq-hosted LLM inference.

**Live demo:** https://gaurav-rag-chatbot.streamlit.app

## Stack

- **Retrieval**: FAISS vector store, `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **Generation**: Groq API (`openai/gpt-oss-20b`), via an OpenAI-compatible interface
- **Orchestration**: LangChain (LCEL retrieval chain)
- **Frontend**: Streamlit
- **Deployed on**: Streamlit Community Cloud

The current corpus is a single document (`data/ai-report.pdf`, the U.S.
Dept. of Education's May 2023 report on AI in education), but the
pipeline is corpus-agnostic — swap the PDF and rebuild the index to
point it at something else.

## Evaluation

Retrieval quality and answer groundedness were measured against a
16-question hand-built eval set, comparing two chunking strategies.
Full methodology, results, and honest limitations are in
[`EVAL_RESULTS.md`](EVAL_RESULTS.md) — short version: the deployed
default (500-char chunks, 50 overlap) beats the alternative (1000/200)
on content-level retrieval recall (62% vs. 50% at k=4), and 15/16
generated answers passed an LLM-as-judge groundedness check.

## Running locally

```bash
git clone https://github.com/random-gau/rag-chatbot1.1.git
cd rag-chatbot1.1
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit .env and add your own GROQ_API_KEY
streamlit run app.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

The repo already includes a prebuilt FAISS index (`faiss_index/`), so
the app runs immediately. To rebuild it (e.g. after changing the PDF
or chunking parameters):

```bash
python3 build_index.py --chunk-size 500 --chunk-overlap 50 --out faiss_index
```

## Repo structure

```
.
├── app.py                  # Streamlit app: retrieval + generation
├── build_index.py          # Builds the FAISS index from data/*.pdf
├── data/                   # Source PDF(s)
├── faiss_index/            # Prebuilt vector index
├── eval/                   # Retrieval + groundedness evaluation harness
│   ├── eval_set.py          # 16-question hand-built eval set with verified gold answers
│   ├── eval_retrieval.py    # Page-level and content-level recall@k
│   └── eval_groundedness.py # LLM-as-judge groundedness check
├── EVAL_RESULTS.md         # Full evaluation writeup
├── requirements.txt
└── .env.example
```

## Notes on some decisions made along the way

- **Groq instead of OpenAI**: originally built against OpenAI, switched
  after running out of free credits. Groq's free tier hosts open-weight
  models via an OpenAI-compatible API, so the change was a small code
  swap, not a rewrite.
- **Streamlit Community Cloud instead of Hugging Face Spaces**: HF
  changed its free-tier policy to require a paid plan for Streamlit/
  Docker-based Spaces. Streamlit Community Cloud is free and needed no
  code changes.
