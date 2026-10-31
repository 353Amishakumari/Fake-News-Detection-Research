import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class FusionModel:
    def __init__(self): self.model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
    def fit(self,X,y): self.model.fit(np.asarray(X),np.asarray(y)); return self
    def predict_proba(self,X): return self.model.predict_proba(np.asarray(X))[:,1]
    def predict(self,X): return self.model.predict(np.asarray(X))
