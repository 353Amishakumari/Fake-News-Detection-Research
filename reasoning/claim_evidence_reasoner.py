
class ClaimEvidenceReasoner:
    """
    Claim-Evidence Reasoning Module.

    Separates:
      1. Evidence relevance
      2. Evidence stance
      3. Final reasoning verdict

    IMPORTANT:
    Evidence stance must come from a claim-specific
    stance classifier or verified annotation.

    Retrieval scores alone cannot determine stance.
    """

    def __init__(
        self,
        min_relevance=0.20,
        min_stance_confidence=0.60
    ):
        self.min_relevance = min_relevance
        self.min_stance_confidence = min_stance_confidence

    def reason(self, claim, ranked_evidence):

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must not be empty.")

        support = 0.0
        contradiction = 0.0
        relevant_count = 0
        assessed_count = 0

        evidence_details = []

        for row in ranked_evidence or []:

            item = dict(row)

            # Relevance score
            relevance = float(
                item.get(
                    "rank_score",
                    item.get(
                        "rerank_score",
                        item.get("score", 0.0)
                    )
                )
            )

            relevance = max(0.0, min(1.0, relevance))

            if relevance < self.min_relevance:
                item["reasoning_status"] = "LOW_RELEVANCE"
                evidence_details.append(item)
                continue

            relevant_count += 1

            # Stance must be claim-specific
            stance = str(
                item.get(
                    "stance_label",
                    item.get("label", "UNASSESSED")
                )
            ).upper().strip()

            confidence = float(
                item.get("stance_confidence", 0.0)
            )

            confidence = max(0.0, min(1.0, confidence))

            if (
                stance not in {"SUPPORT", "CONTRADICT"}
                or confidence < self.min_stance_confidence
            ):
                item["reasoning_status"] = "UNASSESSED"
                evidence_details.append(item)
                continue

            assessed_count += 1

            weighted_score = relevance * confidence

            if stance == "SUPPORT":
                support += weighted_score
            elif stance == "CONTRADICT":
                contradiction += weighted_score

            item["reasoning_status"] = "ASSESSED"
            evidence_details.append(item)

        known = support + contradiction

        support_score = support / known if known else 0.0
        contradiction_score = (
            contradiction / known if known else 0.0
        )

        # Fraction of retrieved evidence that passed
        # the relevance filter. This is not factual confidence.
        availability = (
            relevant_count / len(ranked_evidence)
            if ranked_evidence else 0.0
        )

        if known == 0:
            verdict = "INSUFFICIENT_EVIDENCE"

        elif support_score > contradiction_score:
            verdict = "SUPPORT"

        elif contradiction_score > support_score:
            verdict = "CONTRADICT"

        else:
            verdict = "MIXED"

        return {
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "evidence_availability": availability,
            "relevant_evidence_count": relevant_count,
            "assessed_evidence_count": assessed_count,
            "verdict": verdict,
            "evidence_details": evidence_details
        }


# Backward-compatible function
def reason(claim, ranked_evidence):

    reasoner = ClaimEvidenceReasoner()

    return reasoner.reason(
        claim,
        ranked_evidence
    )
