import os
import sys
import json
import html
import traceback
import io
import hashlib
import re

import streamlit as st
import pandas as pd
import networkx as nx


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Evidence-Grounded Fake News Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
* { box-sizing: border-box; }
.stApp { background:#f3f6fb !important; color:#111827 !important; }
.block-container { max-width:1450px !important; padding-top:2rem !important; padding-bottom:3rem !important; }
.hero { background:linear-gradient(135deg,#111827,#374151) !important; color:#fff !important; padding:2rem !important; border-radius:20px !important; margin-bottom:1.5rem !important; box-shadow:0 8px 24px rgba(0,0,0,.12) !important; }
.hero h1,.hero p { color:#fff !important; }
.hero p { color:#e5e7eb !important; }
.section { color:#111827 !important; font-size:1.35rem !important; font-weight:800 !important; margin:1.6rem 0 .9rem !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, [data-testid="stWidgetLabel"] p { color:#111827 !important; opacity:1 !important; }
textarea { color:#111827 !important; background:#fff !important; border:1px solid #9ca3af !important; }
textarea::placeholder { color:#6b7280 !important; opacity:1 !important; }
[data-testid="stMetric"] { background:#fff !important; border:1px solid #d1d5db !important; border-radius:14px !important; padding:16px !important; box-shadow:0 2px 8px rgba(0,0,0,.05) !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] span { color:#4b5563 !important; opacity:1 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *, [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] span { color:#111827 !important; opacity:1 !important; font-weight:800 !important; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color:#374151 !important; opacity:1 !important; }
.card { background:#fff !important; color:#111827 !important; border:1px solid #d1d5db !important; border-radius:14px !important; padding:1rem 1.15rem !important; margin-bottom:.75rem !important; box-shadow:0 2px 8px rgba(0,0,0,.05) !important; }
.card * { color:#111827 !important; }
section[data-testid="stSidebar"] { background:#fff !important; }
section[data-testid="stSidebar"] * { color:#111827 !important; }
[data-testid="stSlider"] *, [data-testid="stCheckbox"] * { color:#111827 !important; }
[data-testid="stAlert"] * { opacity:1 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:#4b5563 !important; }
.graph-container { background:#fff !important; border:1px solid #d1d5db !important; border-radius:14px !important; padding:12px !important; }
.stButton > button, .stDownloadButton > button { font-weight:700 !important; border-radius:9px !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from retrieval.evidence_retriever import EvidenceRetriever
from retrieval.evidence_reranker import EvidenceReranker
from reasoning.claim_evidence_reasoner import reason
from llm.grounded_explainer import template_explanation, ollama_explanation


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clean_display_text(value):
    """Remove HTML markup and decode entities in retrieved text."""
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_evidence(items):
    """
    Converts different evidence formats into one common format
    used throughout the application.
    """

    output = []

    for i, item in enumerate(items or []):

        page = item.get(
            "page_id",
            item.get("source", "unknown")
        )

        sentence_id = item.get(
            "sentence_id",
            i
        )

        text = clean_display_text(
            item.get(
                "evidence_sentence",
                item.get("text", "")
            )
        )

        output.append(
            {
                "evidence_id": f"{page}_{sentence_id}",
                "label": item.get("label", "UNKNOWN"),
                "source": page,
                "page_id": page,
                "sentence_id": sentence_id,
                "text": text,
                "evidence_sentence": text,

                "semantic_score": safe_float(
                    item.get(
                        "semantic_score",
                        item.get("score", 0.0)
                    )
                ),

                "lexical_score": safe_float(
                    item.get("lexical_score", 0.0)
                ),

                "rerank_score": safe_float(
                    item.get(
                        "rerank_score",
                        item.get("score", 0.0)
                    )
                ),

                "rank": item.get(
                    "rank",
                    i + 1
                ),
            }
        )

    return output


# ============================================================

# ============================================================
# CLASSIFIER
# ============================================================

import joblib

CLASSIFIER_ARTIFACT = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "fake_news_classifier.joblib"
)


@st.cache_resource(show_spinner=False)
def load_existing_classifier():
    """
    Load the pre-trained WELFake classifier artifact.
    """

    if not os.path.exists(CLASSIFIER_ARTIFACT):
        raise FileNotFoundError(
            f"Classifier artifact not found: {CLASSIFIER_ARTIFACT}"
        )

    bundle = joblib.load(CLASSIFIER_ARTIFACT)

    model = bundle.get("model")
    vectorizer = bundle.get("vectorizer")

    if model is None or vectorizer is None:
        raise ValueError(
            "Invalid classifier artifact: model/vectorizer missing."
        )

    return model, vectorizer


def get_classifier():
    """
    Load the exported classifier.
    """

    model, vectorizer = load_existing_classifier()

    return model, vectorizer, "exported_artifact"


def predict(claim):

    try:
        model, vectorizer, source = get_classifier()

        X = vectorizer.transform([claim])

        prediction_value = model.predict(X)[0]

        probabilities = (
            model.predict_proba(X)[0]
            if hasattr(model, "predict_proba")
            else []
        )

        confidence = (
            float(max(probabilities))
            if len(probabilities)
            else 0.0
        )

        # WELFake convention:
        # 0 = FAKE
        # 1 = REAL

        label = (
            "FAKE"
            if int(prediction_value) == 0
            else "REAL"
        )

        return (
            label,
            confidence,
            model,
            vectorizer,
            source,
            None,
        )

    except Exception:
        return (
            "UNKNOWN",
            0.0,
            None,
            None,
            None,
            traceback.format_exc(),
        )


# ============================================================
# LIME
# ============================================================

# ============================================================
# LIME
# ============================================================

def run_lime(
    claim,
    model,
    vectorizer
):

    if (
        model is None
        or vectorizer is None
    ):
        return (
            None,
            "Classifier model/vectorizer unavailable."
        )

    try:

        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(
            class_names=["FAKE", "REAL"]
        )

        def predict_proba(texts):

            X = vectorizer.transform(texts)

            return model.predict_proba(X)

        explanation = explainer.explain_instance(
            claim,
            predict_proba,
            num_features=min(
                12,
                max(
                    3,
                    len(claim.split())
                )
            ),
        )

        return (
            explanation.as_list(),
            None
        )

    except Exception as e:

        return (
            None,
            f"LIME unavailable: {e}"
        )


# ============================================================
# FAITHFULNESS
# ============================================================

def run_faithfulness(
    claim,
    model,
    vectorizer
):

    if (
        model is None
        or vectorizer is None
        or not hasattr(
            model,
            "predict_proba"
        )
    ):
        return (
            None,
            "Classifier model/vectorizer unavailable."
        )

    try:

        original_probability = float(
            max(
                model.predict_proba(
                    vectorizer.transform(
                        [claim]
                    )
                )[0]
            )
        )

        tokens = claim.split()

        if len(tokens) <= 1:
            masked_claim = claim
        else:
            masked_claim = " ".join(
                tokens[
                    :max(
                        1,
                        len(tokens) // 2
                    )
                ]
            )

        masked_probability = float(
            max(
                model.predict_proba(
                    vectorizer.transform(
                        [masked_claim]
                    )
                )[0]
            )
        )

        probability_change = abs(
            original_probability
            - masked_probability
        )

        return (
            {
                "original_probability":
                    original_probability,

                "masked_probability":
                    masked_probability,

                "probability_change":
                    probability_change,
            },
            None,
        )

    except Exception as e:

        return (
            None,
            str(e)
        )


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

def build_kg(
    claim,
    selected
):

    try:

        from knowledge_graph.graph_builder import (
            NewsKnowledgeGraph
        )

        builder = NewsKnowledgeGraph()

        evidence_rows = []

        for item in selected:

            evidence_rows.append(
                {
                    "evidence_id":
                        item["evidence_id"],

                    "label":
                        item["label"],

                    "source":
                        item["source"],

                    "text":
                        item["text"],
                }
            )

        graph = builder.build(
            claim,
            evidence_rows
        )

        return graph, None

    except Exception:

        return (
            None,
            traceback.format_exc()
        )


def display_graph(
    claim,
    selected
):

    """
    Application-level visualization:

        CLAIM -> RETRIEVED EVIDENCE

    This visualization is intentionally kept separate from
    the actual graph generated by graph_builder.py.
    """

    lines = [
        "digraph G {",
        "rankdir=LR;",
        'graph [bgcolor="white", pad="0.3"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="#111827"];',
        'edge [fontname="Arial", fontcolor="#374151"];',
        '"CLAIM" [label="Claim", fillcolor="#dbeafe"];',
    ]

    for i, item in enumerate(
        selected,
        start=1
    ):

        source = str(
            item.get(
                "source",
                "unknown"
            )
        )

        label = (
            source[:35]
            + "\\n"
            + str(
                item.get(
                    "label",
                    "UNKNOWN"
                )
            )
        )

        label = (
            label
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        evidence_id = f"E{i}"

        lines.append(
            f'"{evidence_id}" '
            f'[label="{label}", '
            f'fillcolor="#ecfdf5"];'
        )

        lines.append(
            f'"CLAIM" -> "{evidence_id}" '
            f'[label="retrieved"];'
        )

    lines.append("}")

    dot = "\n".join(lines)

    st.markdown(
        '<div class="graph-container">',
        unsafe_allow_html=True
    )

    try:

        st.graphviz_chart(
            dot,
            use_container_width=True
        )

    except Exception:

        st.warning(
            "Graphviz rendering is unavailable. "
            "Showing the graph structure as a table."
        )

        graph_rows = []

        for i, item in enumerate(
            selected,
            start=1
        ):

            graph_rows.append(
                {
                    "From": "CLAIM",
                    "To": f"E{i}",
                    "Source": item.get(
                        "source",
                        "unknown"
                    ),
                    "Label": item.get(
                        "label",
                        "UNKNOWN"
                    ),
                    "Relation": "retrieved",
                }
            )

        st.dataframe(
            pd.DataFrame(graph_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# REPORT CHART HELPERS
# ============================================================

def create_report_charts(claim, prediction, confidence, reasoning, selected,
                         faithfulness, lime_result, model=None,
                         vectorizer=None, actual_graph=None):
    """Create in-memory PNG charts for the PDF; no files are written."""
    charts = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO

        def save_fig(fig, key):
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            buffer.seek(0)
            charts[key] = buffer

        # Classifier probabilities, when available.
        labels, values = ["FAKE", "REAL"], [0.0, 0.0]
        if model is not None and vectorizer is not None and hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorizer.transform([claim]))[0]
            classes = list(getattr(model, "classes_", range(len(probs))))
            labels, values = [], []
            for cls, prob in zip(classes, probs):
                labels.append("FAKE" if int(cls) == 0 else "REAL")
                values.append(float(prob))
        else:
            labels = [prediction, "Other"]
            values = [float(confidence), max(0.0, 1.0 - float(confidence))]
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.bar(labels, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted probability")
        ax.set_title("Text classifier probabilities")
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, "classifier")

        # Evidence reranking scores.
        if selected:
            names = [f"E{i}: {str(x.get('source', 'source'))[:24]}" for i, x in enumerate(selected, 1)]
            scores = [safe_float(x.get("rerank_score")) for x in selected]
            fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.42 * len(names))))
            ax.barh(names[::-1], scores[::-1])
            ax.set_xlabel("Reranking score")
            ax.set_title("Retrieved evidence ranking")
            ax.grid(axis="x", alpha=0.25)
            save_fig(fig, "reranking")

        # Support / contradiction scores.
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        names = ["Support", "Contradiction", "Availability"]
        vals = [safe_float(reasoning.get("support_score")),
                safe_float(reasoning.get("contradiction_score")),
                safe_float(reasoning.get("evidence_availability"))]
        ax.bar(names, vals)
        ax.set_ylim(0, max(1.0, max(vals, default=0) * 1.15))
        ax.set_title("Claim-evidence reasoning scores")
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, "reasoning")

        # Actual knowledge graph, not the synthetic claim-to-evidence graph.
        if actual_graph is not None and actual_graph.number_of_nodes() > 0:
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            pos = nx.spring_layout(actual_graph, seed=42)
            nx.draw_networkx(actual_graph, pos=pos, ax=ax, with_labels=True,
                             node_size=900, font_size=6, arrows=True)
            ax.set_title("Actual knowledge graph")
            ax.axis("off")
            save_fig(fig, "knowledge_graph")

        # LIME feature contributions.
        if lime_result:
            lime_items = sorted(lime_result, key=lambda x: abs(safe_float(x[1])))
            terms = [str(x[0]) for x in lime_items]
            weights = [safe_float(x[1]) for x in lime_items]
            fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.32 * len(terms))))
            ax.barh(terms, weights)
            ax.axvline(0, linewidth=0.8)
            ax.set_xlabel("LIME contribution")
            ax.set_title("LIME feature importance")
            ax.grid(axis="x", alpha=0.25)
            save_fig(fig, "lime")

        # Faithfulness comparison.
        if faithfulness:
            fig, ax = plt.subplots(figsize=(6.4, 3.2))
            ax.bar(["Original", "Masked"],
                   [safe_float(faithfulness.get("original_probability")),
                    safe_float(faithfulness.get("masked_probability"))])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Faithfulness: original vs masked input")
            ax.grid(axis="y", alpha=0.25)
            save_fig(fig, "faithfulness")

    except Exception:
        # Keep PDF generation working even if plotting is unavailable.
        return {}
    return charts


# ============================================================
# PDF REPORT
# ============================================================

def make_pdf(
    claim,
    prediction,
    confidence,
    reasoning,
    selected,
    faithfulness,
    lime_result,
    model=None,
    vectorizer=None,
    actual_graph=None
):

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as PDFImage,
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=14,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
    )

    story = []
    chart_buffers = create_report_charts(
        claim, prediction, confidence, reasoning, selected,
        faithfulness, lime_result, model, vectorizer, actual_graph
    )

    story.append(
        Paragraph(
            "Evidence-Grounded Fake News Detection Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Research analysis report",
            styles["Heading2"],
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            f"<b>Claim:</b> "
            f"{html.escape(claim)}",
            styles["BodyText"],
        )
    )

    story.append(
        Spacer(1, 8)
    )

    data = [
        [
            "Classifier Prediction",
            prediction,
        ],
        [
            "Confidence",
            f"{confidence * 100:.2f}%",
        ],
        [
            "Reasoning Verdict",
            reasoning.get(
                "verdict",
                "UNKNOWN"
            ),
        ],
        [
            "Support Score",
            f"{safe_float(reasoning.get('support_score')):.3f}",
        ],
        [
            "Contradiction Score",
            f"{safe_float(reasoning.get('contradiction_score')):.3f}",
        ],
        [
            "Evidence Availability",
            f"{safe_float(reasoning.get('evidence_availability')):.3f}",
        ],
    ]

    table = Table(
        data,
        colWidths=[
            2.2 * inch,
            4.3 * inch,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.whitesmoke,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    story.extend([table, Spacer(1, 12)])

    story.append(Paragraph("Research Visualizations", styles["Heading2"]))
    chart_titles = [
        ("classifier", "Classifier probabilities"),
        ("reranking", "Evidence reranking scores"),
        ("reasoning", "Claim-evidence reasoning scores"),
        ("knowledge_graph", "Actual knowledge graph"),
        ("lime", "LIME feature contributions"),
        ("faithfulness", "Faithfulness comparison"),
    ]
    for chart_key, chart_title in chart_titles:
        chart_buffer = chart_buffers.get(chart_key)
        if chart_buffer is not None:
            story.append(Paragraph(chart_title, styles["Heading3"]))
            story.append(PDFImage(chart_buffer, width=6.4 * inch, height=3.0 * inch))
            story.append(Spacer(1, 8))
    if not chart_buffers:
        story.append(Paragraph(
            "Charts could not be generated in this environment. "
            "Install matplotlib to include report visualizations.",
            small_style
        ))

    story.extend(
        [
            Spacer(1, 8),
            Paragraph(
                "Retrieved Evidence",
                styles["Heading2"],
            ),
        ]
    )

    for i, item in enumerate(
        selected,
        start=1
    ):

        story.append(
            Paragraph(
                f"<b>Evidence {i}</b> | "
                f"{html.escape(str(item['label']))} | "
                f"Source: "
                f"{html.escape(str(item['source']))} | "
                f"Rerank: "
                f"{item['rerank_score']:.4f}",
                small_style,
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    clean_display_text(item["text"])
                ),
                small_style,
            )
        )

        story.append(
            Spacer(1, 7)
        )

    story.append(
        Paragraph(
            "Faithfulness",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            html.escape(
                json.dumps(
                    faithfulness
                    or {
                        "status":
                            "not available"
                    },
                    indent=2,
                )
            ),
            small_style,
        )
    )

    story.append(
        Paragraph(
            "LIME",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            html.escape(
                json.dumps(
                    lime_result
                    or {
                        "status":
                            "not available"
                    },
                    indent=2,
                )
            ),
            small_style,
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <h1>🧠 Evidence-Grounded Fake News Detection</h1>
    <p>
        Evidence Retrieval + Reranking + Claim-Evidence Reasoning
        + Knowledge Graph + Explainable AI
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research Settings")

    top_k = st.slider(
        "Number of retrieved evidence items",
        min_value=3,
        max_value=10,
        value=6,
    )

    use_ollama = st.checkbox(
        "Use Ollama for explanation",
        value=False,
    )

    st.markdown("---")

    st.markdown(
        """
**Pipeline**

1. Claim Input
2. Text Classification
3. Semantic Retrieval
4. Evidence Reranking
5. Claim-Evidence Reasoning
6. Knowledge Graph
7. LIME
8. Faithfulness
9. Grounded Explanation
10. Research Metrics
11. PDF Report
"""
    )


# ============================================================
# CLAIM INPUT
# ============================================================

st.markdown(
    '<div class="section">📝 Enter a Claim</div>',
    unsafe_allow_html=True,
)

claim = st.text_area(
    "News claim / statement",
    placeholder=(
        "Example: The Boston Celtics play "
        "their home games at the TD Garden."
    ),
    height=120,
)

analyze = st.button(
    "🔍 Analyze Claim",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not claim.strip():

        st.warning(
            "Please enter a claim first."
        )

        st.stop()

    claim = claim.strip()

    # --------------------------------------------------------
    # 1. CLASSIFIER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">1️⃣ Text Classifier Result</div>',
        unsafe_allow_html=True,
    )

    (
        prediction,
        confidence,
        model,
        vectorizer,
        classifier_source,
        classifier_error,
    ) = predict(claim)

    col1, col2 = st.columns(2)

    col1.metric(
        "Classifier Prediction",
        prediction,
    )

    col2.metric(
        "Model Confidence",
        f"{confidence * 100:.2f}%",
    )

    st.info(
        "Classifier confidence is a model prediction, "
        "not independent factual verification."
    )

    if classifier_source:

        st.caption(
            f"Classifier source: {classifier_source}"
        )

    if classifier_error:

        st.warning(
            "Classifier could not be loaded/trained."
        )

        st.code(
            classifier_error
        )

    # --------------------------------------------------------
    # 2. RETRIEVAL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">2️⃣ Evidence Retrieval</div>',
        unsafe_allow_html=True,
    )

    try:

        with st.spinner(
            "Retrieving evidence..."
        ):

            retriever = EvidenceRetriever()

            retrieved = retriever.retrieve(
                claim,
                top_k=top_k,
            )

        st.success(
            f"Retrieved {len(retrieved)} evidence items."
        )

    except Exception:

        retrieved = []

        st.error(
            "Evidence retrieval failed."
        )

        st.code(
            traceback.format_exc()
        )

    # --------------------------------------------------------
    # 3. RERANKING
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">3️⃣ Evidence Reranking</div>',
        unsafe_allow_html=True,
    )

    try:

        reranker = EvidenceReranker(
            semantic_weight=0.70,
            lexical_weight=0.30,
        )

        reranked = reranker.rerank(
            claim,
            retrieved,
        )

    except Exception:

        reranked = retrieved

        st.warning(
            "Reranking failed; "
            "using semantic retrieval results."
        )

    selected = normalize_evidence(
        reranked
    )[:top_k]

    # --------------------------------------------------------
    # 4. EVIDENCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">4️⃣ Retrieved Evidence</div>',
        unsafe_allow_html=True,
    )

    if not selected:

        st.warning(
            "No evidence was retrieved."
        )

    for i, item in enumerate(selected, start=1):

        evidence_text = clean_display_text(item.get("text", ""))

        with st.container(border=True):
            st.markdown(
                f"**Evidence {i}** | Label: **{html.escape(str(item['label']))}** "
                f"| Source: **{html.escape(str(item['source']))}** "
                f"| Rerank: **{safe_float(item.get('rerank_score')):.4f}**"
            )
            st.write(evidence_text or "No evidence text available.")

    # --------------------------------------------------------
    # 5. REASONING
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">5️⃣ Claim-Evidence Reasoning</div>',
        unsafe_allow_html=True,
    )

    reasoning_rows = []

    for item in selected:

        raw_label = str(
            item["label"]
        ).upper()

        mapped_label = {
            "SUPPORTS": "SUPPORT",
            "REFUTES": "CONTRADICT",
            "SUPPORT": "SUPPORT",
            "CONTRADICT": "CONTRADICT",
        }.get(
            raw_label,
            "UNKNOWN",
        )

        reasoning_rows.append(
            {
                "evidence_id":
                    item["evidence_id"],

                "label":
                    mapped_label,

                "rank_score":
                    item["rerank_score"],

                "source":
                    item["source"],

                "text":
                    clean_display_text(item["text"]),
            }
        )

    try:

        reasoning = reason(
            claim,
            reasoning_rows,
        )

    except Exception:

        reasoning = {
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "evidence_availability": 0.0,
            "verdict": "INSUFFICIENT_EVIDENCE",
        }

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Support",
        f"{safe_float(reasoning.get('support_score')):.2f}",
    )

    c2.metric(
        "Contradiction",
        f"{safe_float(reasoning.get('contradiction_score')):.2f}",
    )

    c3.metric(
        "Evidence Availability",
        f"{safe_float(reasoning.get('evidence_availability')):.2f}",
    )

    c4.metric(
        "Verdict",
        reasoning.get(
            "verdict",
            "UNKNOWN"
        ),
    )

    # --------------------------------------------------------
    # 6. KNOWLEDGE GRAPH
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">6️⃣ Knowledge Graph</div>',
        unsafe_allow_html=True,
    )

    actual_graph, kg_error = build_kg(
        claim,
        selected,
    )

    if actual_graph is not None:

        st.success(
            "Knowledge graph generation completed."
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Actual KG Nodes",
            actual_graph.number_of_nodes(),
        )

        c2.metric(
            "Actual KG Edges",
            actual_graph.number_of_edges(),
        )

        if (
            actual_graph.number_of_edges()
            == 0
        ):

            st.info(
                "The current KG builder found no "
                "entity-to-evidence edges. "
                "The visualization below shows the "
                "claim-to-retrieved-evidence structure "
                "used by the application; it does not "
                "change the underlying KG."
            )

        display_graph(
            claim,
            selected,
        )

        # Visualize the graph produced by graph_builder.py as well.
        if actual_graph.number_of_nodes() > 0:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(actual_graph, seed=42)
                nx.draw_networkx(
                    actual_graph, pos=pos, ax=ax, with_labels=True,
                    node_size=1100, font_size=7, arrows=True
                )
                ax.set_title("Actual Knowledge Graph")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as graph_plot_error:
                st.warning(f"Actual KG plot unavailable: {graph_plot_error}")

        with st.expander(
            "View actual KG data"
        ):

            st.json(
                {
                    "nodes":
                        actual_graph.number_of_nodes(),

                    "edges":
                        actual_graph.number_of_edges(),

                    "node_list":
                        list(
                            actual_graph.nodes()
                        )[:50],
                }
            )

    else:

        st.warning(
            "Knowledge graph could not be generated."
        )

        st.code(
            kg_error
            or "Unknown KG error"
        )

    # --------------------------------------------------------
    # 7. LIME
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">7️⃣ LIME Explanation</div>',
        unsafe_allow_html=True,
    )

    lime_result, lime_error = run_lime(
        claim,
        model,
        vectorizer,
    )

    if lime_result:

        lime_df = pd.DataFrame(
            lime_result,
            columns=[
                "Token",
                "Weight",
            ],
        )

        st.dataframe(
            lime_df,
            use_container_width=True,
            hide_index=True,
        )
        try:
            st.bar_chart(lime_df.set_index("Token")["Weight"])
        except Exception as lime_plot_error:
            st.caption(f"LIME chart unavailable: {lime_plot_error}")

    else:

        st.warning(
            "LIME explanation was not available."
        )

        st.caption(
            lime_error
        )

    # --------------------------------------------------------
    # 8. FAITHFULNESS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">8️⃣ Explanation Faithfulness</div>',
        unsafe_allow_html=True,
    )

    faithfulness, faith_error = run_faithfulness(
        claim,
        model,
        vectorizer,
    )

    if faithfulness:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Original Probability",
            f"{faithfulness['original_probability']:.4f}",
        )

        c2.metric(
            "Masked Probability",
            f"{faithfulness['masked_probability']:.4f}",
        )

        c3.metric(
            "Probability Change",
            f"{faithfulness['probability_change']:.4f}",
        )
        st.bar_chart(pd.DataFrame({
            "Probability": [
                faithfulness["original_probability"],
                faithfulness["masked_probability"]
            ]
        }, index=["Original", "Masked"]))

    else:

        st.warning(
            "Faithfulness evaluation was not available."
        )

        st.caption(
            faith_error
        )

    # --------------------------------------------------------
    # 9. GROUNDED EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">9️⃣ Evidence-Grounded Explanation</div>',
        unsafe_allow_html=True,
    )

    explanation_evidence = []

    for item in selected:

        explanation_evidence.append(
            {
                "label":
                    item["label"],

                "source":
                    item["source"],

                "text":
                    item["text"],

                "score":
                    item["rerank_score"],
            }
        )

    try:

        explanation = template_explanation(
            prediction,
            confidence,
            reasoning,
            explanation_evidence,
        )

    except Exception:

        explanation = (
            "Evidence-grounded explanation "
            "could not be generated."
        )

    if use_ollama:

        prompt = (
            "Analyze the claim using ONLY the supplied evidence. "
            "Do not invent facts.\n\n"
            f"Claim: {claim}\n"
            f"Prediction: {prediction}\n"
            f"Reasoning: {json.dumps(reasoning)}\n"
            "Evidence:\n"
        )

        for i, item in enumerate(
            explanation_evidence,
            start=1
        ):

            prompt += (
                f"{i}. "
                f"[{item['label']}] "
                f"{item['text']} "
                f"(source: {item['source']})\n"
            )

        try:

            ollama_output = ollama_explanation(
                prompt
            )

            if ollama_output:

                explanation = ollama_output

        except Exception:

            pass

    with st.container(border=True):
        st.write(clean_display_text(explanation))

    # --------------------------------------------------------
    # 10. RESEARCH METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">🔬 Retrieval Research Metrics</div>',
        unsafe_allow_html=True,
    )

    reranked_metrics_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "reranked_retrieval_metrics.csv",
    )

    baseline_metrics_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "retrieval_metrics.csv",
    )

    if os.path.exists(
        reranked_metrics_path
    ):

        try:

            st.dataframe(
                pd.read_csv(
                    reranked_metrics_path
                ),
                use_container_width=True,
                hide_index=True,
            )

        except Exception:

            pass

    if (
        os.path.exists(
            reranked_metrics_path
        )
        and os.path.exists(
            baseline_metrics_path
        )
    ):

        try:

            baseline_df = pd.read_csv(
                baseline_metrics_path
            )

            reranked_df = pd.read_csv(
                reranked_metrics_path
            )

            comparison_rows = []

            for metric in [
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "MRR",
            ]:

                baseline_value = baseline_df.loc[
                    baseline_df.metric == metric,
                    "value",
                ]

                reranked_value = reranked_df.loc[
                    reranked_df.metric == metric,
                    "value",
                ]

                if (
                    not baseline_value.empty
                    and not reranked_value.empty
                ):

                    b = float(
                        baseline_value.iloc[0]
                    )

                    r = float(
                        reranked_value.iloc[0]
                    )

                    comparison_rows.append(
                        {
                            "Metric":
                                metric,

                            "Semantic Retrieval":
                                b,

                            "Semantic + Lexical Reranking":
                                r,

                            "Improvement (pp)":
                                (r - b) * 100,
                        }
                    )

            if comparison_rows:

                st.dataframe(
                    pd.DataFrame(
                        comparison_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception:

            pass

    # --------------------------------------------------------
    # 11. PDF
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">📄 Download Analysis PDF</div>',
        unsafe_allow_html=True,
    )

    try:

        pdf_bytes = make_pdf(
            claim,
            prediction,
            confidence,
            reasoning,
            selected,
            faithfulness,
            lime_result,
            model=model,
            vectorizer=vectorizer,
            actual_graph=actual_graph,
        )

        st.download_button(
            "📥 Download PDF Report",
            data=pdf_bytes,
            file_name="fake_news_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception:

        st.error(
            "PDF generation failed."
        )

        st.code(
            traceback.format_exc()
        )

    # --------------------------------------------------------
    # 12. INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">📊 Research Interpretation</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The system separates text classification, "
        "evidence retrieval, evidence reranking, "
        "claim-evidence reasoning, knowledge-graph "
        "construction, explainability, and grounded "
        "explanation. A classifier prediction is not "
        "treated as independent factual verification."
    )
