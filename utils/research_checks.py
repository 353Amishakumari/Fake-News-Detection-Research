import json
from pathlib import Path

def save_json(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=str))


def check_evidence_integrity(rows):
    """Flags missing provenance or unsupported labels before research metrics are computed."""
    problems=[]
    for r in rows:
        if not r.get('source'): problems.append(f"{r.get('evidence_id')}: missing source")
        if r.get('label') not in {'SUPPORT','CONTRADICT','UNKNOWN'}: problems.append(f"{r.get('evidence_id')}: invalid label")
    return problems
