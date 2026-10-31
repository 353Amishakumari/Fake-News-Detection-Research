import sys
import os
import pandas as pd

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.evidence_reranker import EvidenceReranker


# ============================================================
# PATHS
# ============================================================

CLAIMS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "fever",
    "fever_claims_1000.csv"
)

EVIDENCE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "fever",
    "fever_evidence_1000.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "reranked_retrieval_metrics.csv"
)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def recall_at_k(results, gold_evidence, k):
    retrieved = set(
        (item["page_id"], int(item["sentence_id"]))
        for item in results[:k]
    )

    return 1.0 if retrieved.intersection(gold_evidence) else 0.0


def reciprocal_rank(results, gold_evidence):
    for rank, item in enumerate(results, start=1):

        evidence_pair = (
            item["page_id"],
            int(item["sentence_id"])
        )

        if evidence_pair in gold_evidence:
            return 1.0 / rank

    return 0.0


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("RERANKED EVIDENCE RETRIEVAL EVALUATION")
print("=" * 70)

print("\nChecking dataset files...")

if not os.path.exists(CLAIMS_PATH):
    raise FileNotFoundError(
        f"Claims file not found: {CLAIMS_PATH}"
    )

if not os.path.exists(EVIDENCE_PATH):
    raise FileNotFoundError(
        f"Evidence file not found: {EVIDENCE_PATH}"
    )

print("Dataset files found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading FEVER datasets...")

claims_df = pd.read_csv(CLAIMS_PATH)
evidence_df = pd.read_csv(EVIDENCE_PATH)

print(f"Claims loaded   : {len(claims_df)}")
print(f"Evidence loaded : {len(evidence_df)}")


# ============================================================
# BUILD GOLD EVIDENCE MAPPING
# ============================================================

print("\nBuilding gold evidence mapping...")

gold_mapping = {}

for _, row in evidence_df.iterrows():

    claim_id = row["claim_id"]

    if pd.isna(row["page_id"]) or pd.isna(row["sentence_id"]):
        continue

    evidence_pair = (
        str(row["page_id"]),
        int(row["sentence_id"])
    )

    if claim_id not in gold_mapping:
        gold_mapping[claim_id] = set()

    gold_mapping[claim_id].add(evidence_pair)


print(
    f"Claims with gold evidence: {len(gold_mapping)}"
)


# ============================================================
# SELECT CLAIMS
# ============================================================

evaluation_claims = claims_df[
    claims_df["claim_id"].isin(gold_mapping.keys())
].copy()

print(
    f"\nClaims selected for evaluation: "
    f"{len(evaluation_claims)}"
)


# ============================================================
# INITIALIZE RETRIEVER
# ============================================================

print("\nInitializing Evidence Retriever...")

retriever = EvidenceRetriever(
    evidence_path=EVIDENCE_PATH
)

reranker = EvidenceReranker()

print("\nRetriever and reranker ready.")


# ============================================================
# EVALUATION
# ============================================================

recall1_scores = []
recall5_scores = []
recall10_scores = []
mrr_scores = []

total = len(evaluation_claims)

print("\nRunning reranked retrieval evaluation...")

for index, (_, row) in enumerate(
    evaluation_claims.iterrows(),
    start=1
):

    claim = str(row["claim"])
    claim_id = row["claim_id"]

    # --------------------------------------------------------
    # Step 1: Semantic retrieval
    # --------------------------------------------------------

    retrieved = retriever.retrieve(
        claim,
        top_k=10
    )

    # --------------------------------------------------------
    # Step 2: Evidence reranking
    # --------------------------------------------------------

    reranked = reranker.rerank(
        claim,
        retrieved
    )

    gold_evidence = gold_mapping[claim_id]

    # --------------------------------------------------------
    # Step 3: Metrics
    # --------------------------------------------------------

    recall1_scores.append(
        recall_at_k(
            reranked,
            gold_evidence,
            1
        )
    )

    recall5_scores.append(
        recall_at_k(
            reranked,
            gold_evidence,
            5
        )
    )

    recall10_scores.append(
        recall_at_k(
            reranked,
            gold_evidence,
            10
        )
    )

    mrr_scores.append(
        reciprocal_rank(
            reranked,
            gold_evidence
        )
    )

    if index % 100 == 0 or index == total:
        print(
            f"Evaluated {index}/{total} claims"
        )


# ============================================================
# FINAL METRICS
# ============================================================

recall1 = sum(recall1_scores) / total
recall5 = sum(recall5_scores) / total
recall10 = sum(recall10_scores) / total
mrr = sum(mrr_scores) / total


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("RERANKED RETRIEVAL RESULTS")
print("=" * 70)

print(f"\nRecall@1  : {recall1:.4f}")
print(f"Recall@1  : {recall1 * 100:.2f}%")

print(f"\nRecall@5  : {recall5:.4f}")
print(f"Recall@5  : {recall5 * 100:.2f}%")

print(f"\nRecall@10 : {recall10:.4f}")
print(f"Recall@10 : {recall10 * 100:.2f}%")

print(f"\nMRR       : {mrr:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "method": "Semantic + Lexical Reranking",
        "recall_at_1": recall1,
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "mrr": mrr,
        "num_claims": total
    }
])

metrics_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nMetrics saved successfully:")
print(OUTPUT_PATH)

print("\nEvaluation finished.")