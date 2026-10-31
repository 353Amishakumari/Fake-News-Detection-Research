import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EvidenceRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder=SentenceTransformer(model_name)
        self.df=None; self.embeddings=None

    def fit(self,evidence_df):
        self.df=evidence_df.reset_index(drop=True)
        self.embeddings=self.encoder.encode(self.df.text.tolist(),normalize_embeddings=True,show_progress_bar=False)
        return self

    def retrieve(self,claim,top_k=5):
        if self.df is None: raise RuntimeError('Retriever is not fitted.')
        q=self.encoder.encode([claim],normalize_embeddings=True)
        scores=cosine_similarity(q,self.embeddings)[0]
        idx=np.argsort(scores)[::-1][:top_k]
        rows=[]
        for i in idx:
            row=self.df.iloc[int(i)].to_dict(); row['similarity']=float(scores[i]); rows.append(row)
        return rows
