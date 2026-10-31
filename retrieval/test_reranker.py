import os
import sys

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT MODULES
# ============================================================

from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.evidence_reranker import EvidenceReranker


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EVIDENCE RETRIEVAL + RERANKING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Test claim
    # --------------------------------------------------------

    claim = (
        "The Boston Celtics play their home games at TD Garden."
    )

    print("\nClaim:")
    print(claim)

    # --------------------------------------------------------
    # STEP 2: Initialize Evidence Retriever
    # --------------------------------------------------------

    print("\n[1] Initializing Evidence Retriever...")

    evidence_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "fever",
        "fever_evidence_1000.csv"
    )

    retriever = EvidenceRetriever(
        evidence_path=evidence_path
    )

    # --------------------------------------------------------
    # STEP 3: Retrieve evidence
    # --------------------------------------------------------

    print("\n[2] Retrieving Top-10 evidence...")

    retrieved = retriever.retrieve(
        claim,
        top_k=10
    )

    print(
        f"Retrieved {len(retrieved)} evidence items."
    )

    # --------------------------------------------------------
    # STEP 4: Initialize Reranker
    # --------------------------------------------------------

    print("\n[3] Initializing Evidence Reranker...")

    reranker = EvidenceReranker()

    # --------------------------------------------------------
    # STEP 5: Rerank evidence
    # --------------------------------------------------------

    print("\n[4] Re-ranking evidence...")

    reranked = reranker.rerank(
        claim,
        retrieved
    )

    # --------------------------------------------------------
    # STEP 6: Original retrieval results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ORIGINAL SEMANTIC RETRIEVAL")
    print("=" * 70)

    for item in retrieved[:5]:

        print("\n----------------------------------------")

        print(
            f"Rank: {item['rank']}"
        )

        print(
            f"Similarity Score: "
            f"{item['score']:.4f}"
        )

        print(
            f"Page: "
            f"{item['page_id']}"
        )

        print(
            f"Sentence ID: "
            f"{item['sentence_id']}"
        )

        print("\nEvidence:")

        print(
            item["evidence_sentence"]
        )

    # --------------------------------------------------------
    # STEP 7: Reranked results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RERANKED EVIDENCE")
    print("=" * 70)

    for item in reranked[:5]:

        print("\n----------------------------------------")

        print(
            f"New Rank: "
            f"{item['rank']}"
        )

        print(
            f"Semantic Score: "
            f"{item['semantic_score']:.4f}"
        )

        print(
            f"Lexical Score: "
            f"{item['lexical_score']:.4f}"
        )

        print(
            f"Final Rerank Score: "
            f"{item['rerank_score']:.4f}"
        )

        print(
            f"Page: "
            f"{item['page_id']}"
        )

        print(
            f"Sentence ID: "
            f"{item['sentence_id']}"
        )

        print("\nEvidence:")

        print(
            item["evidence_sentence"]
        )

    # --------------------------------------------------------
    # STEP 8: Completed
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RERANKING TEST COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()