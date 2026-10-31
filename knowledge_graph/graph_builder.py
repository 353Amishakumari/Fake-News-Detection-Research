import re
import networkx as nx
try:
    import spacy
except ImportError:
    spacy=None

class NewsKnowledgeGraph:
    def __init__(self, model='en_core_web_sm'):
        if spacy is None: raise ImportError('Install spaCy and en_core_web_sm.')
        try: self.nlp=spacy.load(model)
        except OSError as e: raise RuntimeError('Run: python -m spacy download en_core_web_sm') from e

    def build(self, article, evidence_rows=None):
        doc=self.nlp(article)
        g=nx.MultiDiGraph()
        ents=[e for e in doc.ents if e.label_ in {'PERSON','ORG','GPE','LOC','EVENT','PRODUCT','DATE','LAW','NORP'}]
        for e in ents: g.add_node(e.text, type=e.label_)
        for sent in doc.sents:
            sents_ents=[e.text for e in sent.ents if e.text in g.nodes]
            for i in range(len(sents_ents)):
                for j in range(i+1,len(sents_ents)):
                    g.add_edge(sents_ents[i],sents_ents[j],relation='co-mentioned',sentence=sent.text[:300])
        if evidence_rows:
            for r in evidence_rows:
                eid=f"EVIDENCE:{r['evidence_id']}"
                g.add_node(eid,type='EVIDENCE',label=r.get('label','UNKNOWN'),source=r.get('source',''))
                for e in ents:
                    if re.search(r'\b'+re.escape(e.text)+r'\b',r['text'],re.I):
                        g.add_edge(e.text,eid,relation=str(r.get('label','UNKNOWN')).lower(),evidence=r['text'][:300])
        return g
