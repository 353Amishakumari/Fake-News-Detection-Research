def mask_words(text, words):
    out=text
    for w in words: out=out.replace(w,'')
    return out

def evaluate_lime_faithfulness(text, features, predict_proba, top_k=10):
    chosen=[x['feature'] for x in features[:top_k]]
    before=float(predict_proba([text])[0][1])
    after_text=mask_words(text,chosen)
    after=float(predict_proba([after_text])[0][1])
    return {'top_k':top_k,'original_real_probability':before,'masked_real_probability':after,'probability_change':abs(before-after),'removed_features':chosen}
