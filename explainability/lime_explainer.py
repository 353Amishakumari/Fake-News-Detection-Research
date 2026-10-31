from lime.lime_text import LimeTextExplainer

class LIMEExplainer:
    def __init__(self, class_names=('FAKE','REAL')):
        self.explainer=LimeTextExplainer(class_names=list(class_names),random_state=42)

    def explain(self, text, predict_proba, num_features=12):
        exp=self.explainer.explain_instance(text,predict_proba,num_features=num_features)
        return [{'feature':w,'weight':float(v)} for w,v in exp.as_list()]
