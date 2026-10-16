from pathlib import Path

from chunker import load_documents, chunk_documents
from bm25 import build_bm25_index
from retriever import hybrid_retrieve, multi_query_retrieve
from reranker import rerank
from query_rewriter import rewrite_query
from generator import generate_answer

SCRIPT_DIR = Path(__file__).parent

# Your ground-truth test set: each entry is a question paired with a
# substring you KNOW exists in sample.txt and directly answers it.
TEST_CASES = [
    [
        {"question": "What did Ali work on at TechnoGenics?",
         "expected_substring": "StrikeReady, an enterprise cybersecurity"},
        {"question": "By how much did he reduce API response time there?",
         "expected_substring": "Reduced average API response time by 30%"},
    ],
    [
        {"question": "What is PORTpass?",
         "expected_substring": "PORTpass, a patient vaccination records platform"},
        {"question": "How much did it improve data retrieval efficiency by?",
         "expected_substring": "improved data retrieval efficiency by 20%"},
    ],
    [
        {"question": "Tell me about the company's remote work policy.",
         "expected_substring": "up to 4 days per week"},
        {"question": "Is there any stipend related to it?",
         "expected_substring": "$300 stipend"},
    ],
]

def evaluate(test_cases, chunks, metadatas, bm25_index, retrieve_fn, top_k=3):
    """Run each conversation through the pipeline using retrieve_fn, checking retrieval rank at every turn."""
    results = []

    for conversation in test_cases:
        history = []
        for turn in conversation:
            question = turn["question"]
            expected = turn["expected_substring"]

            standalone_question = rewrite_query(question, history)

            candidates = retrieve_fn(standalone_question, chunks, metadatas, bm25_index, top_k=15)
            reranked = rerank(standalone_question, candidates, top_k=top_k)

            rank = None
            for i, (chunk, _score, _source) in enumerate(reranked, start=1):
                if expected.lower() in chunk.lower():
                    rank = i
                    break

            results.append({"question": question, "rewritten": standalone_question, "rank": rank})

            # A real generated answer — not the expected_substring — goes into
            # history, so later turns in this conversation get rewritten
            # against the same context a live run would actually produce.
            answer = generate_answer(standalone_question, reranked)
            history.append((question, answer))

    return results


def summarize(results):
    """Compute Hit Rate@k and MRR from per-question rank results."""
    hits = [r for r in results if r["rank"] is not None]
    hit_rate = len(hits) / len(results)
    mrr = sum(1 / r["rank"] for r in hits) / len(results)
    return hit_rate, mrr


if __name__ == "__main__":
    documents = load_documents(SCRIPT_DIR / "documents")
    chunks, metadatas = chunk_documents(documents)
    bm25_index = build_bm25_index(chunks)

    retrievers = [
        ("hybrid_retrieve", hybrid_retrieve),
        ("multi_query_retrieve", multi_query_retrieve),
    ]

    for name, retrieve_fn in retrievers:
        print(f"=== {name} ===")
        results = evaluate(TEST_CASES, chunks, metadatas, bm25_index, retrieve_fn)

        for r in results:
            status = f"rank {r['rank']}" if r["rank"] else "MISS"
            print(f"[{status}] {r['question']}  ->  {r['rewritten']}")

        hit_rate, mrr = summarize(results)
        print(f"Hit Rate@3: {hit_rate:.2%}")
        print(f"MRR: {mrr:.3f}\n")