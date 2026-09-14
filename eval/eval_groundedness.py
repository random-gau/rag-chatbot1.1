"""
Groundedness check via LLM-as-judge.

For each eval question: retrieve context, generate an answer (same
prompt/model as the deployed app), then ask the model a second time,
in a fresh call, to judge whether every claim in the answer is actually
supported by the retrieved context.

IMPORTANT LIMITATION -- read before you quote this number anywhere:
this is a self-consistency check, not verified accuracy. The judge is
the same underlying model (just a separate call), so it can share the
generator's blind spots, and it can be wrong. Report this as "N/16
answers judged grounded by an LLM-as-judge check," not as "N/16
answers are correct" -- those are different claims and the second one
overstates what this script measures.

Usage:
    python eval_groundedness.py ../faiss_index
Requires GROQ_API_KEY in the environment (loads ../.env automatically).
"""
import sys
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, ".")
from eval_set import EVAL_SET

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 4

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are checking whether an answer is fully supported by the given "
    "context. Respond with exactly one word, PASS or FAIL, on the first "
    "line, then a one-sentence reason on the second line.\n\n"
    "PASS means either (a) every factual claim in the answer is directly "
    "supported by the context, or (b) the answer honestly states it "
    "doesn't know / the context doesn't contain the answer, AND the "
    "context genuinely does not contain the requested information. "
    "FAIL means the answer contains any claim not supported by the "
    "context (including plausible-sounding elaboration not actually "
    "present in the text), OR the answer says it doesn't know while the "
    "context actually does contain the answer (that would be an "
    "unnecessary refusal, a different kind of failure worth flagging "
    "but still not 'grounded' in the sense being measured -- treat it "
    "as FAIL and say so in your reason).\n\n"
    "Context:\n{context}\n\nAnswer to check:\n{answer}"
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_groundedness.py <faiss_index_path>")
        sys.exit(1)
    index_path = sys.argv[1]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    results = []
    for item in EVAL_SET:
        docs = db.similarity_search(item["question"], k=RETRIEVER_K)
        context = "\n\n".join(d.page_content for d in docs)

        answer_msg = llm.invoke(
            ANSWER_PROMPT.format_messages(context=context, question=item["question"])
        )
        answer = answer_msg.content

        judge_msg = llm.invoke(
            JUDGE_PROMPT.format_messages(context=context, answer=answer)
        )
        judge_text = judge_msg.content.strip()
        verdict = "PASS" if judge_text.upper().startswith("PASS") else "FAIL"

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "verdict": verdict,
                "judge_reason": judge_text,
            }
        )
        print(f"[{verdict}] Q{item['id']}: {item['question']}")

    passed = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n{passed}/{len(results)} answers judged grounded by LLM-as-judge "
          f"(self-consistency check, not verified ground truth).")

    failures = [r for r in results if r["verdict"] == "FAIL"]
    if failures:
        print("\n--- FAIL details ---")
        for r in failures:
            print(f"Q{r['id']}: {r['question']}")
            print(f"  answer: {r['answer'][:200]}")
            print(f"  judge:  {r['judge_reason']}")


if __name__ == "__main__":
    main()
