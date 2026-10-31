import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def build_lr():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])


def build_svm():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000, sublinear_tf=True)),
        ('clf', LinearSVC(class_weight='balanced'))
    ])


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
