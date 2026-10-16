from pathlib import Path

from chunker import load_documents, chunk_documents
from bm25 import build_bm25_index
from retriever import hybrid_retrieve
from reranker import rerank

SCRIPT_DIR = Path(__file__).parent

# Your ground-truth test set: each entry is a question paired with a
# substring you KNOW exists in sample.txt and directly answers it.
TEST_CASES = [
    {"question": "What backend frameworks does Ali know?",
     "expected_substring": "Django REST Framework, FastAPI, Flask"},

    {"question": "What databases has Ali worked with?",
     "expected_substring": "PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch"},

    {"question": "What frontend technologies does Ali use?",
     "expected_substring": "React, Next.js, Angular, TypeScript"},

    {"question": "What AI-related technologies has Ali worked with?",
     "expected_substring": "Model Context Protocol (MCP)"},

    {"question": "What did Ali work on at TechnoGenics?",
     "expected_substring": "StrikeReady, an enterprise cybersecurity"},

    {"question": "By how much did Ali reduce API response time at TechnoGenics?",
     "expected_substring": "Reduced average API response time by 30%"},

    {"question": "What did Ali build at Merik Solutions?",
     "expected_substring": "enterprise healthcare management system on Django REST Framework"},

    {"question": "What kind of systems did Ali work on at Rapidev?",
     "expected_substring": "large-scale operational and planning systems"},

    {"question": "What is PORTpass?",
     "expected_substring": "PORTpass, a patient vaccination records platform"},

    {"question": "What chatbot did Ali build at Enigmatix?",
     "expected_substring": "WhatsApp chatbot using Django and the Twilio API"},

    {"question": "What e-commerce project has Ali built?",
     "expected_substring": "Trendoes"},

    {"question": "What is Ali's educational background?",
     "expected_substring": "Master of Computer Science (MCS)"},

    {"question": "How many days per week can employees work remotely?",
     "expected_substring": "up to 4 days per week"},

    {"question": "How much is the home office equipment stipend?",
     "expected_substring": "$300 stipend"},

    {"question": "How many PTO days do employees accrue per month?",
     "expected_substring": "1.5 days of PTO per month"},
]


def evaluate(test_cases, chunks, metadatas, bm25_index, top_k=3):
    """Run the retrieval pipeline on each test case and score how well it found the right chunk."""
    results = []

    for case in test_cases:
        question = case["question"]
        expected = case["expected_substring"]

        candidates = hybrid_retrieve(question, chunks, metadatas, bm25_index, top_k=15)
        reranked = rerank(question, candidates, top_k=top_k)

        # Find the rank (1-based position) of the first chunk containing
        # the expected substring; None if it's not in the top_k at all.
        rank = None
        for i, (chunk, _score, _source) in enumerate(reranked, start=1):
            if expected.lower() in chunk.lower():
                rank = i
                break

        results.append({"question": question, "rank": rank})

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

    results = evaluate(TEST_CASES, chunks, metadatas, bm25_index)

    for r in results:
        status = f"rank {r['rank']}" if r["rank"] else "MISS"
        print(f"[{status}] {r['question']}")

    hit_rate, mrr = summarize(results)
    print(f"\nHit Rate@3: {hit_rate:.2%}")
    print(f"MRR: {mrr:.3f}")