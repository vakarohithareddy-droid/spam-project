"""
train_model.py
Trains a Multinomial Naive Bayes spam classifier on dataset.csv using
scikit-learn, evaluates it, and saves the fitted pipeline (vectorizer +
classifier) to model.pkl for the Flask backend to load.
"""
import csv
import pickle
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

texts, labels = [], []
with open("dataset.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row["text"])
        labels.append(1 if row["label"] == "spam" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

clf = MultinomialNB(alpha=1.0)
clf.fit(Xtr, y_train)

pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, pred).tolist()  # [[TN FP][FN TP]]
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"False positive rate: {false_positive_rate:.4f}")
print(f"Confusion matrix (rows=true, cols=pred) [[TN FP][FN TP]]: {cm}")

with open("model.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)

metrics = {
    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc,
    "false_positive_rate": false_positive_rate,
    "confusion_matrix": cm, "confusion_labels": ["ham", "spam"],
    "n_train": len(X_train), "n_test": len(X_test), "vocab_size": len(vectorizer.vocabulary_),
    "dataset_source": "UCI SMS Spam Collection (5,572 real labeled messages)",
    "n_spam": int(sum(y_train) + sum(y_test)), "n_ham": int(len(labels) - (sum(y_train) + sum(y_test))),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model.pkl and metrics.json")
