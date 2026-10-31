"""External generalization evaluation.

Usage:
python evaluation/cross_dataset.py --model artifacts/lr.joblib --external data/external.csv
External CSV needs text,label.
"""
import argparse,joblib,pandas as pd
from utils.data_utils import load_welfake
from utils.metrics import classification_metrics

p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--external',required=True); args=p.parse_args()
model=joblib.load(args.model); df=load_welfake(args.external); pred=model.predict(df.text)
prob=model.predict_proba(df.text)[:,1] if hasattr(model,'predict_proba') else None
print(classification_metrics(df.label,pred,prob))
