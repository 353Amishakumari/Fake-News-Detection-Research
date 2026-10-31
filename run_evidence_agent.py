from agents.evidence_agent import EvidenceRetrievalAgent
from retrieval.evidence_retriever import EvidenceRetriever


def main():

    print("=" * 70)
    print("EVIDENCE RETRIEVAL AGENT")
    print("=" * 70)

    # Create the agent
    agent = EvidenceRetrievalAgent()

    # Test claim
    claim = "The Boston Celtics play their home games at TD Garden."

    print("\nClaim:")
    print(claim)

    print("\nRunning evidence retrieval agent...\n")

    # Run agent
    result = agent.run(
        claim=claim,
        top_k=5
    )

    print("Status:")
    print(result["status"])

    print("\nRetrieved Evidence:")
    print("-" * 70)

    evidence = result["retrieved_evidence"]

    if not evidence:
        print("No evidence found.")
        return

    for item in evidence:

        print(f"\nRank: {item.get('rank')}")
        print(f"Similarity Score: {item.get('score'):.4f}")
        print(f"Page: {item.get('page_id')}")
        print(f"Sentence ID: {item.get('sentence_id')}")

        print("Evidence:")
        print(item.get("evidence_sentence"))

        print("-" * 70)


if __name__ == "__main__":
    main()