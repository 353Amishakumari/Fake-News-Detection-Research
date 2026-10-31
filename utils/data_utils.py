import re
import pandas as pd


def clean_text(x):
    x = '' if pd.isna(x) else str(x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()


def normalize_label(x):
    s = str(x).strip().lower()
    if s in {'fake','false','0','f'}: return 0
    if s in {'real','true','1','r'}: return 1
    try: return int(float(s))
    except Exception: raise ValueError(f'Unknown label: {x}')


def load_welfake(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get('text')
    if text_col is None:
        title_col = cols.get('title')
        if title_col is None: raise ValueError('Need text or title column.')
        df['text'] = df[title_col].fillna('')
    else:
        df['text'] = df[text_col].fillna('')
        title_col = cols.get('title')
        if title_col: df['text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    label_col = cols.get('label')
    if label_col is None: raise ValueError('Need label column.')
    df['label'] = df[label_col].map(normalize_label)
    df['text'] = df['text'].map(clean_text)
    return df[['text','label']].dropna().reset_index(drop=True)


def load_evidence(path):
    df = pd.read_csv(path)
    required = {'evidence_id','text','source','label'}
    missing = required - set(c.lower().strip() for c in df.columns)
    if missing: raise ValueError(f'Evidence CSV missing columns: {missing}')
    rename = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=rename)
    df['text'] = df['text'].map(clean_text)
    df['label'] = df['label'].astype(str).str.upper().str.strip()
    return df[['evidence_id','text','source','label']].dropna().reset_index(drop=True)
