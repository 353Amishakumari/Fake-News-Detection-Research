import os, requests

def template_explanation(prediction, confidence, reasoning, evidence):
    top=evidence[:3]
    if not top:
        return ('Evidence-grounded explanation unavailable because no evidence items were supplied. '
                'The classifier prediction should therefore be treated as model-based, not externally verified.')
    lines=[f"The classifier predicts {prediction} with confidence {confidence:.2f}.",
           f"Evidence reasoning: {reasoning['verdict']}.",
           f"Support score: {reasoning['support_score']:.2f}; contradiction score: {reasoning['contradiction_score']:.2f}.",
           "Top retrieved evidence:"]
    for i,e in enumerate(top,1):
        lines.append(f"{i}. [{e.get('label','UNKNOWN')}] {e['text']} (source: {e.get('source','unknown')})")
    return ' '.join(lines)

def ollama_explanation(prompt, model=None):
    model=model or os.getenv('OLLAMA_MODEL','llama3.2')
    try:
        r=requests.post('http://localhost:11434/api/generate',json={'model':model,'prompt':prompt,'stream':False},timeout=60)
        r.raise_for_status(); return r.json().get('response','')
    except Exception:
        return None
