from pathlib import Path
import json, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from utils.data_utils import load_welfake, load_evidence
from reasoning.fusion import FusionModel

# Research prototype: trains a fusion model from article semantics + evidence statistics.
# It requires evidence metadata aligned to each training article for a scientifically valid run.
# For safety, this script does NOT invent evidence labels.

DATA='data/welfake.csv'; EVID='data/evidence.csv'; OUT=Path('artifacts'); OUT.mkdir(exist_ok=True)

def main():
    df=load_welfake(DATA)
    if not Path(EVID).exists():
        raise FileNotFoundError('Create data/evidence.csv with legitimate evidence before training the proposed evidence model.')
    ev=load_evidence(EVID)
    enc=SentenceTransformer('all-MiniLM-L6-v2')
    # Evidence corpus is global; article-level alignment must be supplied by the researcher.
    # This script demonstrates the feature interface and trains only if evidence_index.csv exists.
    align=Path('data/evidence_index.csv')
    if not align.exists():
        raise FileNotFoundError('Create data/evidence_index.csv with columns: article_id,evidence_id,support_score,contradiction_score,availability.')
    idx=__import__('pandas').read_csv(align)
    required={'article_id','support_score','contradiction_score','availability'}
    if not required.issubset(idx.columns): raise ValueError(f'Missing {required-set(idx.columns)}')
    # article_id must correspond to row index in WELFake.
    df['article_id']=range(len(df))
    merged=df.merge(idx,on='article_id',how='inner')
    if len(merged)<100: raise ValueError('Too few aligned training examples; build a proper evidence mapping.')
    emb=enc.encode(merged.text.tolist(),normalize_embeddings=True,show_progress_bar=True)
    X=np.hstack([emb,merged[['support_score','contradiction_score','availability']].to_numpy()])
    y=merged.label.to_numpy()
    model=FusionModel().fit(X,y)
    joblib.dump({'model':model,'encoder_name':'all-MiniLM-L6-v2'},OUT/'fusion_model.joblib')
    Path(OUT/'proposed_training_meta.json').write_text(json.dumps({'n':len(merged),'embedding_dim':emb.shape[1]},indent=2))
    print('Saved artifacts/fusion_model.joblib')

if __name__=='__main__': main()
