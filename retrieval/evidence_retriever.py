
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class EvidenceRetriever:

    def __init__(
        self,
        evidence_path="data/fever/fever_evidence_1000.csv",
        model_name="all-MiniLM-L6-v2",
        semantic_weight=0.70,
        lexical_weight=0.30,
    ):

        print("=" * 70)
        print("INITIALIZING HYBRID EVIDENCE RETRIEVER")
        print("=" * 70)

        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        # Resolve path relative to project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        if not os.path.isabs(evidence_path):
            evidence_path = os.path.join(
                project_root,
                evidence_path
            )

        if not os.path.exists(evidence_path):
            raise FileNotFoundError(
                f"Evidence dataset not found: {evidence_path}"
            )

        print("\nLoading FEVER evidence dataset...")

        self.df = pd.read_csv(evidence_path)

        required_columns = [
            "evidence_sentence",
            "page_id",
            "sentence_id",
        ]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required CSV columns: {missing}"
            )

        self.df = self.df.dropna(
            subset=["evidence_sentence"]
        ).copy()

        self.df["evidence_sentence"] = (
            self.df["evidence_sentence"]
            .astype(str)
            .str.strip()
        )

        self.df = self.df[
            self.df["evidence_sentence"].str.len() > 10
        ]

        self.df = self.df.drop_duplicates(
            subset=["evidence_sentence"]
        ).reset_index(drop=True)

        if self.df.empty:
            raise ValueError(
                "No valid evidence sentences found."
            )

        print(
            f"Evidence sentences loaded: {len(self.df)}"
        )

        # ----------------------------------------------------
        # SEMANTIC EMBEDDINGS
        # ----------------------------------------------------

        print("\nLoading sentence-transformer model...")

        self.model = SentenceTransformer(model_name)

        print("Creating semantic embeddings...")

        self.evidence_embeddings = self.model.encode(
            self.df["evidence_sentence"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        print("Semantic embeddings created.")

        # ----------------------------------------------------
        # LEXICAL INDEX
        # ----------------------------------------------------

        print("\nCreating TF-IDF lexical index...")

        self.tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )

        self.evidence_tfidf = self.tfidf.fit_transform(
            self.df["evidence_sentence"].tolist()
        )

        print("TF-IDF index created.")

        print("=" * 70)
        print("HYBRID EVIDENCE RETRIEVER READY")
        print("=" * 70)

    def retrieve(self, claim, top_k=5):

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must be a non-empty string.")

        claim = claim.strip()
        top_k = max(1, int(top_k))

        # ----------------------------------------------------
        # 1. SEMANTIC SIMILARITY
        # ----------------------------------------------------

        claim_embedding = self.model.encode(
            [claim],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        semantic_scores = cosine_similarity(
            claim_embedding,
            self.evidence_embeddings
        )[0]

        # ----------------------------------------------------
        # 2. LEXICAL SIMILARITY
        # ----------------------------------------------------

        claim_tfidf = self.tfidf.transform([claim])

        lexical_scores = cosine_similarity(
            claim_tfidf,
            self.evidence_tfidf
        )[0]

        # ----------------------------------------------------
        # 3. HYBRID SCORE
        # ----------------------------------------------------

        hybrid_scores = (
            self.semantic_weight * semantic_scores
            + self.lexical_weight * lexical_scores
        )

        # Retrieve a larger candidate pool before selection
        candidate_count = min(
            len(self.df),
            max(top_k * 5, top_k)
        )

        candidate_indices = np.argsort(
            hybrid_scores
        )[::-1][:candidate_count]

        results = []
        seen_sentences = set()

        # ----------------------------------------------------
        # 4. BUILD RESULTS
        # ----------------------------------------------------

        for index in candidate_indices:

            row = self.df.iloc[index]

            evidence_text = str(
                row["evidence_sentence"]
            ).strip()

            normalized_text = evidence_text.lower()

            if normalized_text in seen_sentences:
                continue

            seen_sentences.add(normalized_text)

            page_id = str(row["page_id"])
            sentence_id = str(row["sentence_id"])

            results.append(
                {
                    "rank": len(results) + 1,

                    "evidence_id": (
                        f"{page_id}_{sentence_id}"
                    ),

                    "claim_id": row.get(
                        "claim_id",
                        None
                    ),

                    "page_id": page_id,

                    "source": page_id,

                    "sentence_id": sentence_id,

                    "evidence_sentence": evidence_text,

                    "text": evidence_text,

                    # IMPORTANT:
                    # Original FEVER labels must not be
                    # treated as stance labels for a new claim.
                    "label": "UNASSESSED",

                    "stance_label": "UNASSESSED",

                    "semantic_score": float(
                        semantic_scores[index]
                    ),

                    "lexical_score": float(
                        lexical_scores[index]
                    ),

                    "rerank_score": float(
                        hybrid_scores[index]
                    ),

                    "score": float(
                        hybrid_scores[index]
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results
