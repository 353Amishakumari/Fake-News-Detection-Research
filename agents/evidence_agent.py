from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from retrieval.evidence_retriever import EvidenceRetriever


class EvidenceAgentState(TypedDict):
    claim: str
    top_k: int
    retrieved_evidence: List[Dict[str, Any]]
    status: str


class EvidenceRetrievalAgent:
    """
    Evidence Retrieval Agent

    Workflow:
    Claim
      ↓
    Evidence Retrieval
      ↓
    Evidence Validation
      ↓
    Top-K Evidence
    """

    def __init__(self):
        # Load the evidence retriever
        self.retriever = EvidenceRetriever()

        # Create LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph workflow.
        """

        workflow = StateGraph(EvidenceAgentState)

        # Add nodes
        workflow.add_node(
            "retrieve_evidence",
            self.retrieve_evidence
        )

        workflow.add_node(
            "validate_evidence",
            self.validate_evidence
        )

        # Define workflow
        workflow.add_edge(
            START,
            "retrieve_evidence"
        )

        workflow.add_edge(
            "retrieve_evidence",
            "validate_evidence"
        )

        workflow.add_edge(
            "validate_evidence",
            END
        )

        return workflow.compile()

    def retrieve_evidence(
        self,
        state: EvidenceAgentState
    ) -> Dict[str, Any]:
        """
        Retrieve relevant evidence for the given claim.
        """

        claim = state["claim"]
        top_k = state.get("top_k", 5)

        try:
            evidence = self.retriever.retrieve(
                claim,
                top_k=top_k
            )

            return {
                "retrieved_evidence": evidence,
                "status": "Evidence retrieved successfully"
            }

        except Exception as e:

            return {
                "retrieved_evidence": [],
                "status": f"Retrieval failed: {str(e)}"
            }

    def validate_evidence(
        self,
        state: EvidenceAgentState
    ) -> Dict[str, Any]:
        """
        Validate and remove duplicate evidence.
        """

        evidence = state.get(
            "retrieved_evidence",
            []
        )

        unique_evidence = []
        seen = set()

        for item in evidence:

            key = (
                item.get("page_id"),
                item.get("sentence_id")
            )

            if key not in seen:

                seen.add(key)
                unique_evidence.append(item)

        # Reassign ranking after removing duplicates
        for rank, item in enumerate(
            unique_evidence,
            start=1
        ):
            item["rank"] = rank

        return {
            "retrieved_evidence": unique_evidence,
            "status": "Evidence validated successfully"
        }

    def run(
        self,
        claim: str,
        top_k: int = 5
    ) -> EvidenceAgentState:
        """
        Run the complete evidence retrieval agent.
        """

        initial_state: EvidenceAgentState = {
            "claim": claim,
            "top_k": top_k,
            "retrieved_evidence": [],
            "status": "Starting agent"
        }

        result = self.graph.invoke(
            initial_state
        )

        return result