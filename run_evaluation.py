from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_utils import load_welfake
from utils.metrics import classification_metrics

ART=Path('artifacts')
df=load_welfake('data/welfake.csv')
Xtr,Xte,ytr,yte=train_test_split(df.text,df.label,test_size=.2,random_state=42,stratify=df.label)
rows=[]
for name in ['lr','svm']:
    path=ART/f'{name}.joblib'
    if not path.exists(): continue
    m=joblib.load(path); pred=m.predict(Xte)
    prob=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else None
    z=classification_metrics(yte,pred,prob); z['model']=name; rows.append(z)
pd.DataFrame(rows).to_csv(ART/'evaluation.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
