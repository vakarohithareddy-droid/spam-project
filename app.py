"""
app.py
Flask backend that serves the trained Naive Bayes spam classifier via a
JSON REST API, and serves the static frontend (index.html).

Run:
    pip install -r requirements.txt
    python train_model.py       # trains model.pkl (run once)
    python app.py
Then open http://localhost:5000

Environment variables:
    FLASK_DEBUG        "1" to enable Flask's debugger/reloader (default: off)
    SPAM_THRESHOLD      float in (0, 1), decision boundary on P(spam) (default: 0.5)
    MAX_TEXT_LENGTH     max characters accepted per request (default: 20000)
"""
from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__, static_folder=".", static_url_path="")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at '{MODEL_PATH}'. "
        "Run `python train_model.py` first to train and save the model."
    )

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
vectorizer = saved["vectorizer"]
clf = saved["clf"]

# clf.classes_ is [0, 1] where 0 = ham, 1 = spam (see train_model.py)
feature_names = vectorizer.get_feature_names_out()
log_prob_ham = clf.feature_log_prob_[0]
log_prob_spam = clf.feature_log_prob_[1]

# Decision threshold on P(spam). Kept configurable rather than silently
# hardcoded, since the "right" threshold is a product decision that trades
# precision against recall (see the confusion matrix in metrics.json) —
# 0.5 is a reasonable, defensible default, not an arbitrary guess.
SPAM_THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.5))

# Guards against pathologically large request bodies driving up latency /
# memory on the vectorizer for no real benefit (mirrors the client-side cap
# in index.html).
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 20000))


def top_reasons(X, label, n=5):
    """Return the n words present in `text` that most strongly pushed the
    prediction toward the winning label, using the same log-likelihood-ratio
    idea shown in the frontend's token highlighting."""
    nonzero = X.nonzero()[1]
    if len(nonzero) == 0:
        return []
    diffs = log_prob_spam[nonzero] - log_prob_ham[nonzero]
    words = feature_names[nonzero]
    order = diffs.argsort()
    if label == "spam":
        top_idx = order[::-1][:n]  # largest positive diff = most spam-leaning
    else:
        top_idx = order[:n]  # most negative diff = most ham-leaning
    return [words[i] for i in top_idx]


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"Text too long ({len(text)} chars). "
                     f"Maximum is {MAX_TEXT_LENGTH} characters."
        }), 400

    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]  # [P(ham), P(spam)]
    probability_spam = float(proba[1])
    probability_ham = float(proba[0])
    label = "spam" if probability_spam >= SPAM_THRESHOLD else "ham"
    confidence = probability_spam if label == "spam" else probability_ham

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "probability_spam": round(probability_spam, 4),
        "probability_ham": round(probability_ham, 4),
        "reason": top_reasons(X, label),
        "threshold": SPAM_THRESHOLD,
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, port=int(os.environ.get("PORT", 5000)))
