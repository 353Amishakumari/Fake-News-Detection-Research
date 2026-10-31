import re

class ClaimExtractor:
    """Lightweight claim extraction baseline.

    For a research paper, compare this heuristic with an NLI/LLM-based claim extractor
    and report the effect. This baseline deliberately avoids inventing claims.
    """
    def extract(self, text, max_claims=5):
        sentences=re.split(r'(?<=[.!?])\s+', str(text).strip())
        claims=[]
        for s in sentences:
            s=s.strip()
            if len(s.split())>=6 and not s.endswith('?'):
                claims.append(s)
            if len(claims)>=max_claims: break
        return claims
