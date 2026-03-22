from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']

df['label'] = df['label'].astype(str).str.lower()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    user_vec = vectorizer.transform([message])
    prediction = model.predict(user_vec)

    if prediction[0] == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return jsonify({'result': result})
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
