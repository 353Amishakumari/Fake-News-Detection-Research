from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score

def classification_metrics(y_true,y_pred,y_prob=None):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    out={'accuracy':accuracy_score(y_true,y_pred),'precision':p,'recall':r,'f1':f}
    if y_prob is not None: out['roc_auc']=roc_auc_score(y_true,y_prob)
    return out
