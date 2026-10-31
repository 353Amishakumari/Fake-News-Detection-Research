# Evidence-Grounded Knowledge Graph Reasoning for Faithful Fake News Detection

Research-oriented upgrade of the original Explainable Fake News Detection project.

## Pipeline
1. WELFake article -> text preprocessing
2. TF-IDF + Logistic Regression / SVM baselines
3. Sentence-Transformer semantic representation
4. Claim extraction
5. Evidence retrieval from a local evidence corpus
6. Evidence ranking
7. Entity/relation Knowledge Graph
8. Claim-evidence reasoning
9. Fusion predictor
10. LIME explanation + faithfulness test
11. Grounded explanation (template fallback; optional Ollama)
12. Streamlit demo

## Important research note
WELFake itself does not provide external evidence for each article. Therefore this implementation separates:
- `data/welfake.csv`: classification dataset
- `data/evidence.csv`: evidence corpus with `evidence_id,text,source,label` where label is SUPPORT/CONTRADICT/UNKNOWN if available.

Do not report evidence-grounded metrics until a properly sourced evidence corpus is supplied.

## Install
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Prepare data
Put WELFake CSV at `data/welfake.csv`. Expected columns can be `text,label` or `title,text,label`.

Evidence CSV is optional for baseline training, but required for the proposed evidence experiments. Expected columns:
`evidence_id,text,source,label`

## Train
```bash
python train_baselines.py
python train_proposed.py
python run_evaluation.py
```

## App
```bash
streamlit run app.py
```

## Optional Ollama
Set `OLLAMA_MODEL` and make sure Ollama is running locally. If unavailable, the app uses a deterministic grounded template and never invents evidence.
