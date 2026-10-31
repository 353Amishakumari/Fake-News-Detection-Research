from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.data_utils import load_welfake
from models.baselines import build_lr, build_svm

DATA='data/welfake.csv'; OUT=Path('artifacts'); OUT.mkdir(exist_ok=True)
df=load_welfake(DATA)
Xtr,Xte,ytr,yte=train_test_split(df.text,df.label,test_size=.2,random_state=42,stratify=df.label)
results=[]
for name,model in [('lr',build_lr()),('svm',build_svm())]:
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    p,r,f,_=precision_recall_fscore_support(yte,pred,average='binary',zero_division=0)
    results.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':p,'recall':r,'f1':f})
    joblib.dump(model,OUT/f'{name}.joblib')
Path(OUT/'baseline_results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
