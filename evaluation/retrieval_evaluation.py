import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# This allows us to import:
# retrieval.evidence_retriever
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import pandas as pd
import numpy as np

from retrieval.evidence_retriever import EvidenceRetriever


# ============================================================
# FILE PATHS
# ============================================================

EVIDENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fever",
    "fever_evidence_1000.csv"
)

CLAIMS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fever",
    "fever_claims_1000.csv"
)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts"
)

OUTPUT_FILE = os.path.join(
    ARTIFACTS_DIR,
    "retrieval_metrics.csv"
)


# ============================================================
# BUILD GOLD EVIDENCE MAPPING
# ============================================================

def build_gold_evidence(evidence_df):
    """
    Create mapping:

        claim_id -> set of gold evidence IDs

    Each evidence ID is:

        (page_id, sentence_id)
    """

    gold = {}

    for _, row in evidence_df.iterrows():

        claim_id = row["claim_id"]

        evidence_id = (
            str(row["page_id"]),
            str(row["sentence_id"])
        )

        if claim_id not in gold:
            gold[claim_id] = set()

        gold[claim_id].add(evidence_id)

    return gold


# ============================================================
# CALCULATE RETRIEVAL METRICS
# ============================================================

def calculate_metrics(results):
    """
    Calculate:

        Recall@1
        Recall@5
        Recall@10
        MRR
    """

    total = len(results)

    if total == 0:

        return {
            "Recall@1": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR": 0.0
        }

    recall_1_count = 0
    recall_5_count = 0
    recall_10_count = 0

    reciprocal_ranks = []

    for result in results:

        retrieved = result["retrieved"]
        gold = result["gold"]

        # ----------------------------------------------------
        # Recall@1
        # ----------------------------------------------------

        top1 = retrieved[:1]

        if any(
            evidence_id in gold
            for evidence_id in top1
        ):

            recall_1_count += 1

        # ----------------------------------------------------
        # Recall@5
        # ----------------------------------------------------

        top5 = retrieved[:5]

        if any(
            evidence_id in gold
            for evidence_id in top5
        ):

            recall_5_count += 1

        # ----------------------------------------------------
        # Recall@10
        # ----------------------------------------------------

        top10 = retrieved[:10]

        if any(
            evidence_id in gold
            for evidence_id in top10
        ):

            recall_10_count += 1

        # ----------------------------------------------------
        # Reciprocal Rank
        # ----------------------------------------------------

        reciprocal_rank = 0.0

        for rank, evidence_id in enumerate(
            retrieved,
            start=1
        ):

            if evidence_id in gold:

                reciprocal_rank = 1.0 / rank

                break

        reciprocal_ranks.append(
            reciprocal_rank
        )

    metrics = {

        "Recall@1":
            recall_1_count / total,

        "Recall@5":
            recall_5_count / total,

        "Recall@10":
            recall_10_count / total,

        "MRR":
            float(
                np.mean(
                    reciprocal_ranks
                )
            )
    }

    return metrics


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("EVIDENCE RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check dataset files
    # --------------------------------------------------------

    print("\nChecking dataset files...")

    if not os.path.exists(EVIDENCE_FILE):

        print(
            f"ERROR: Evidence file not found:\n"
            f"{EVIDENCE_FILE}"
        )

        return

    if not os.path.exists(CLAIMS_FILE):

        print(
            f"ERROR: Claims file not found:\n"
            f"{CLAIMS_FILE}"
        )

        return

    print("Dataset files found.")

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    print("\nLoading FEVER datasets...")

    claims_df = pd.read_csv(
        CLAIMS_FILE
    )

    evidence_df = pd.read_csv(
        EVIDENCE_FILE
    )

    print(
        f"Claims loaded   : {len(claims_df)}"
    )

    print(
        f"Evidence loaded : {len(evidence_df)}"
    )

    # --------------------------------------------------------
    # 3. Build gold evidence mapping
    # --------------------------------------------------------

    print("\nBuilding gold evidence mapping...")

    gold_evidence = build_gold_evidence(
        evidence_df
    )

    print(
        f"Claims with gold evidence: "
        f"{len(gold_evidence)}"
    )

    # --------------------------------------------------------
    # 4. Initialize Evidence Retriever
    # --------------------------------------------------------

    print("\nInitializing Evidence Retriever...")

    retriever = EvidenceRetriever(
        evidence_path=EVIDENCE_FILE
    )

    # --------------------------------------------------------
    # 5. Select claims for evaluation
    # --------------------------------------------------------

    evaluation_claims = claims_df[
        claims_df["claim_id"].isin(
            gold_evidence.keys()
        )
    ].copy()

    print(
        f"\nClaims selected for evaluation: "
        f"{len(evaluation_claims)}"
    )

    # --------------------------------------------------------
    # 6. Run retrieval
    # --------------------------------------------------------

    results = []

    total_claims = len(
        evaluation_claims
    )

    print("\nRunning retrieval evaluation...\n")

    for counter, (_, row) in enumerate(
        evaluation_claims.iterrows(),
        start=1
    ):

        claim_id = row["claim_id"]

        claim = row["claim"]

        # Retrieve Top-10 evidence
        retrieved_results = retriever.retrieve(
            claim,
            top_k=10
        )

        # Convert retrieved evidence
        # into (page_id, sentence_id)
        retrieved_ids = []

        for item in retrieved_results:

            evidence_id = (
                str(item["page_id"]),
                str(item["sentence_id"])
            )

            retrieved_ids.append(
                evidence_id
            )

        # Store result
        results.append(
            {
                "claim_id":
                    claim_id,

                "retrieved":
                    retrieved_ids,

                "gold":
                    gold_evidence[claim_id]
            }
        )

        print(
            f"\rEvaluated "
            f"{counter}/{total_claims} claims",
            end=""
        )

    print("\n\nRetrieval evaluation completed.")

    # --------------------------------------------------------
    # 7. Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    # --------------------------------------------------------
    # 8. Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nRecall@1  : "
        f"{metrics['Recall@1']:.4f}"
    )

    print(
        f"Recall@1  : "
        f"{metrics['Recall@1'] * 100:.2f}%"
    )

    print(
        f"\nRecall@5  : "
        f"{metrics['Recall@5']:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{metrics['Recall@5'] * 100:.2f}%"
    )

    print(
        f"\nRecall@10 : "
        f"{metrics['Recall@10']:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{metrics['Recall@10'] * 100:.2f}%"
    )

    print(
        f"\nMRR       : "
        f"{metrics['MRR']:.4f}"
    )

    print("\n" + "=" * 70)

    # --------------------------------------------------------
    # 9. Save results
    # --------------------------------------------------------

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    output_df = pd.DataFrame(
        [
            {
                "Metric": "Recall@1",
                "Score": metrics["Recall@1"]
            },
            {
                "Metric": "Recall@5",
                "Score": metrics["Recall@5"]
            },
            {
                "Metric": "Recall@10",
                "Score": metrics["Recall@10"]
            },
            {
                "Metric": "MRR",
                "Score": metrics["MRR"]
            }
        ]
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nMetrics saved successfully:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nEvaluation finished.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()