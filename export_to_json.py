"""
export_to_json.py
Exports the fitted TfidfVectorizer + MultinomialNB parameters to a compact
JSON file so the exact same trained model can run inference client-side in
the browser (pure JS, no server needed) for the live demo website.

Because the model is trained on TF-IDF features (not raw counts), the
client-side scorer has to reproduce the same TF-IDF transform: term
frequency * inverse document frequency, then L2-normalized per document,
matching scikit-learn's default TfidfVectorizer behavior.
"""
import pickle
import json

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

vectorizer = saved["vectorizer"]
clf = saved["clf"]

vocab = vectorizer.vocabulary_          # word -> column index
idf = vectorizer.idf_                    # per-feature inverse document frequency
feature_log_prob = clf.feature_log_prob_  # shape (2, n_features): [ham, spam]
class_log_prior = clf.class_log_prior_.tolist()  # [ham, spam]
stop_words = list(vectorizer.get_stop_words()) if vectorizer.get_stop_words() else []

# word -> [idf, log P(word|ham), log P(word|spam)]
word_data = {}
for word, idx in vocab.items():
    word_data[word] = [
        round(float(idf[idx]), 6),
        round(float(feature_log_prob[0][idx]), 6),
        round(float(feature_log_prob[1][idx]), 6),
    ]

export = {
    "class_log_prior": [round(x, 6) for x in class_log_prior],  # [ham, spam]
    "classes": ["ham", "spam"],
    "word_data": word_data,  # word -> [idf, log_prob_ham, log_prob_spam]
    "stop_words": sorted(stop_words),
    "vectorization": "tfidf_l2_normalized",
}

with open("model_params.json", "w") as f:
    json.dump(export, f)

print(f"Exported {len(word_data)} vocabulary words to model_params.json")
print(f"File size: {__import__('os').path.getsize('model_params.json') / 1024:.1f} KB")
