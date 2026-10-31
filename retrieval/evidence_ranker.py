
import re


def tokens(text):
    """
    Convert text into lowercase tokens.
    """
    return set(
        re.findall(r"\b[a-z0-9]+\b", str(text).lower())
    )


def entity_overlap(claim, evidence):
    """
    Calculate lexical overlap.

    Note:
    This is word overlap, not true named-entity recognition.
    """

    claim_tokens = tokens(claim)
    evidence_tokens = tokens(evidence)

    if not claim_tokens:
        return 0.0

    common = claim_tokens.intersection(evidence_tokens)

    return len(common) / len(claim_tokens)


def rank_evidence(claim, rows):
    """
    Rank evidence using semantic similarity
    and lexical overlap.

    This function measures relevance only.
    It does not determine SUPPORTS or REFUTES.
    """

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim must not be empty.")

    ranked = []
    seen = set()

    for row in rows or []:

        item = dict(row)

        # Support both old and new evidence field names
        evidence_text = item.get(
            "evidence_sentence",
            item.get("text", "")
        )

        evidence_text = str(evidence_text).strip()

        if not evidence_text:
            continue

        # Remove duplicate evidence
        key = evidence_text.lower()

        if key in seen:
            continue

        seen.add(key)

        # Read semantic similarity safely
        similarity = float(
            item.get(
                "semantic_score",
                item.get(
                    "similarity",
                    item.get("score", 0.0)
                )
            )
        )

        similarity = max(0.0, min(1.0, similarity))

        # Calculate lexical overlap
        overlap = entity_overlap(
            claim,
            evidence_text
        )

        # Combined relevance score
        score = (
            0.75 * similarity
            + 0.25 * overlap
        )

        item["semantic_score"] = similarity
        item["entity_overlap"] = overlap
        item["rank_score"] = score

        # Preserve stance as unassessed
        item["stance_label"] = "UNASSESSED"
        item["label"] = "UNASSESSED"

        ranked.append(item)

    ranked.sort(
        key=lambda x: x["rank_score"],
        reverse=True
    )

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked
