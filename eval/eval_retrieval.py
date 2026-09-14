"""
Retrieval evaluation: hit-rate@k for one or more FAISS indexes.

A "hit" for a question means at least one of the top-k retrieved chunks
has a source page within +/-1 of any of that question's gold_pages. The
+/-1 tolerance exists because chunk boundaries don't align to page
breaks -- a chunk can start a few lines before or after a page turn, so
penalizing an off-by-one page would understate a retriever that's
actually finding the right passage.

Usage:
    python eval_retrieval.py 500_50=../faiss_index 1000_200=../faiss_index_1000_200
    python eval_retrieval.py --k 2 4 8 500_50=../faiss_index

Each positional arg is label=path to a FAISS index directory (built by
build_index.py). Prints a hit-rate table across the requested k values,
plus a per-question breakdown so you can see *which* questions each
strategy fails, not just an aggregate score.
"""
import argparse
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.path.insert(0, ".")
from eval_set import EVAL_SET

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PAGE_TOLERANCE = 1


def is_page_hit(retrieved_docs, gold_pages, tolerance=PAGE_TOLERANCE):
    """Loose check: did any retrieved chunk come from (near) the right page?
    Generous -- a page can span several chunks, so this can count a hit
    even when the specific retrieved chunk doesn't contain the answer."""
    retrieved_pages = {doc.metadata.get("page") for doc in retrieved_docs}
    for gold in gold_pages:
        for rp in retrieved_pages:
            if rp is not None and abs(rp - gold) <= tolerance:
                return True
    return False


def is_content_hit(retrieved_docs, gold_keywords):
    """Strict check: does any retrieved chunk's actual text contain a
    verified source substring? This is what 'the retriever found the
    answer' should mean -- page-level proximity alone can overcount."""
    for doc in retrieved_docs:
        text = doc.page_content.lower()
        for kw in gold_keywords:
            if kw.lower() in text:
                return True
    return False


def evaluate_index(index_path, k_values):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    max_k = max(k_values)
    # k -> list of (question_id, page_hit_bool, content_hit_bool)
    results = {k: [] for k in k_values}

    for item in EVAL_SET:
        docs = db.similarity_search(item["question"], k=max_k)
        for k in k_values:
            page_hit = is_page_hit(docs[:k], item["gold_pages"])
            content_hit = is_content_hit(docs[:k], item["gold_keywords"])
            results[k].append((item["id"], page_hit, content_hit))

    return results


def print_report(label, results, k_values):
    print(f"\n=== {label} ===")
    for k in k_values:
        page_hits = sum(1 for _, p, _ in results[k] if p)
        content_hits = sum(1 for _, _, c in results[k] if c)
        total = len(results[k])
        print(
            f"  k={k}:  page-level {page_hits}/{total} ({page_hits / total:.0%})"
            f"   |   content-level {content_hits}/{total} ({content_hits / total:.0%})"
        )

    # Per-question breakdown at the largest k, so misses are visible even
    # when they'd be masked by a higher k succeeding.
    max_k = max(k_values)
    content_misses = [qid for qid, _, c in results[max_k] if not c]
    if content_misses:
        print(f"  content-level misses at k={max_k}: question ids {content_misses}")
    else:
        print(f"  no content-level misses at k={max_k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "indexes",
        nargs="+",
        help="One or more label=path pairs, e.g. 500_50=../faiss_index",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[2, 4, 8], help="k values to evaluate"
    )
    args = parser.parse_args()

    all_results = {}
    for spec in args.indexes:
        if "=" not in spec:
            parser.error(f"'{spec}' is not in label=path format")
        label, path = spec.split("=", 1)
        print(f"Evaluating '{label}' at {path} ...")
        results = evaluate_index(path, args.k)
        all_results[label] = results
        print_report(label, results, args.k)

    if len(all_results) > 1:
        print("\n=== Comparison (content-level recall -- the strict, defensible number) ===")
        header = "k".ljust(6) + "".join(l.ljust(16) for l in all_results)
        print(header)
        for k in args.k:
            row = str(k).ljust(6)
            for label in all_results:
                hits = sum(1 for _, _, c in all_results[label][k] if c)
                total = len(all_results[label][k])
                row += f"{hits}/{total} ({hits / total:.0%})".ljust(16)
            print(row)

        print("\n=== Comparison (page-level recall -- more generous, less strict) ===")
        print(header)
        for k in args.k:
            row = str(k).ljust(6)
            for label in all_results:
                hits = sum(1 for _, p, _ in all_results[label][k] if p)
                total = len(all_results[label][k])
                row += f"{hits}/{total} ({hits / total:.0%})".ljust(16)
            print(row)


if __name__ == "__main__":
    main()
