
import re
import math
from collections import Counter


class EvidenceReranker:
    """
    Hybrid Evidence Reranker.

    Signals:
      1. Semantic similarity
      2. IDF-weighted lexical relevance

    Final score:
      alpha * semantic_score + beta * lexical_score

    IMPORTANT:
    Relevance is not the same as stance.
    A high reranking score does not mean SUPPORTS or REFUTES.
    """

    def __init__(
        self,
        semantic_weight=0.70,
        lexical_weight=0.30
    ):

        total = semantic_weight + lexical_weight

        if total <= 0:
            raise ValueError("Reranking weights must sum to > 0.")

        self.semantic_weight = semantic_weight / total
        self.lexical_weight = lexical_weight / total

        self.stop_words = {
            "the", "is", "a", "an", "of", "to", "in",
            "on", "for", "and", "or", "was", "were",
            "are", "as", "at", "by", "from", "with",
            "that", "this", "it", "its", "be", "has",
            "have", "had", "been", "being", "their",
            "they", "them", "he", "she", "his", "her",
            "we", "our", "you", "your", "i", "me",
            "my", "do", "does", "did", "can", "could",
            "would", "should", "will", "shall", "may",
            "might", "must", "not"
        }

    # --------------------------------------------------------
    # TEXT NORMALIZATION
    # --------------------------------------------------------

    def normalize_text(self, text):

        text = str(text).lower()

        tokens = re.findall(
            r"\b[a-z0-9]+\b",
            text
        )

        return [
            token for token in tokens
            if token not in self.stop_words
        ]

    # --------------------------------------------------------
    # LEXICAL RELEVANCE
    # --------------------------------------------------------

    def lexical_overlap(self, claim, evidence):

        claim_tokens = self.normalize_text(claim)
        evidence_tokens = self.normalize_text(evidence)

        if not claim_tokens or not evidence_tokens:
            return 0.0

        claim_counts = Counter(claim_tokens)
        evidence_counts = Counter(evidence_tokens)

        common = sum(
            min(count, evidence_counts[token])
            for token, count in claim_counts.items()
        )

        denominator = sum(claim_counts.values())

        return min(1.0, common / denominator)

    # --------------------------------------------------------
    # IDF-WEIGHTED LEXICAL SCORE
    # --------------------------------------------------------

    def lexical_score(self, claim, evidence):

        claim_tokens = self.normalize_text(claim)
        evidence_tokens = set(
            self.normalize_text(evidence)
        )

        if not claim_tokens or not evidence_tokens:
            return 0.0

        # Inverse frequency within the claim:
        # rare claim terms receive greater weight.
        frequencies = Counter(claim_tokens)

        weights = {
            token: math.log(
                1.0 + len(claim_tokens) / frequency
            )
            for token, frequency in frequencies.items()
        }

        total_weight = sum(weights.values())

        if total_weight == 0:
            return 0.0

        matched_weight = sum(
            weight
            for token, weight in weights.items()
            if token in evidence_tokens
        )

        return min(
            1.0,
            matched_weight / total_weight
        )

    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    def rerank(self, claim, retrieved_evidence):

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must not be empty.")

        reranked = []
        seen = set()

        for item in retrieved_evidence or []:

            new_item = dict(item)

            evidence_text = new_item.get(
                "evidence_sentence",
                new_item.get("text", "")
            )

            evidence_text = str(evidence_text).strip()

            if not evidence_text:
                continue

            # Remove duplicate evidence sentences
            normalized = evidence_text.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            # Use semantic score, not a previously
            # combined hybrid score.
            semantic_score = float(
                new_item.get(
                    "semantic_score",
                    new_item.get("score", 0.0)
                )
            )

            semantic_score = max(
                0.0,
                min(1.0, semantic_score)
            )

            # Calculate lexical relevance
            lexical_score = self.lexical_score(
                claim,
                evidence_text
            )

            overlap_score = self.lexical_overlap(
                claim,
                evidence_text
            )

            # Combined relevance score
            final_score = (
                self.semantic_weight * semantic_score
                + self.lexical_weight * lexical_score
            )

            new_item["semantic_score"] = semantic_score
            new_item["lexical_score"] = lexical_score
            new_item["lexical_overlap"] = overlap_score
            new_item["rerank_score"] = final_score

            # A relevance score must never be treated
            # as a claim-specific stance annotation.
            new_item["label"] = "UNASSESSED"
            new_item["stance_label"] = "UNASSESSED"

            reranked.append(new_item)

        # Highest relevance first
        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank

        return reranked
