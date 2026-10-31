"""Ablation experiment template.

Create one feature table with columns:
text,label,text_score,evidence_support,evidence_contradiction,availability,kg_score
and run the script. It compares feature subsets using identical train/test splits.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

FEATURES={
    'text_only':['text_score'],
    'text_evidence':['text_score','evidence_support','evidence_contradiction','availability'],
    'text_evidence_kg':['text_score','evidence_support','evidence_contradiction','availability','kg_score'],
}

def run(path='data/fusion_features.csv'):
    df=pd.read_csv(path)
    train,test=train_test_split(df,test_size=.2,random_state=42,stratify=df.label)
    rows=[]
    for name,cols in FEATURES.items():
        model=LogisticRegression(max_iter=1000).fit(train[cols],train.label)
        pred=model.predict(test[cols])
        p,r,f,_=precision_recall_fscore_support(test.label,pred,average='binary',zero_division=0)
        rows.append({'variant':name,'accuracy':accuracy_score(test.label,pred),'precision':p,'recall':r,'f1':f})
    out=pd.DataFrame(rows); Path('artifacts').mkdir(exist_ok=True); out.to_csv('artifacts/ablation_results.csv',index=False); print(out)

if __name__=='__main__': run()
