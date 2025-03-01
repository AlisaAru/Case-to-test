import vectorizer
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
#
#
# df = pd.read_csv("labelled_newscatcher_dataset.csv", sep=";", encoding="utf-8")
#
#
# TEXT_COL = "title"
# LABEL_COL = "topic"
#
#
# df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
#
# X = df[TEXT_COL].values
# y = df[LABEL_COL].values
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# vectorizer = TfidfVectorizer(stop_words='english')
# X_train_tfidf = vectorizer.fit_transform(X_train)
#
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_tfidf, y_train)
#
# X_test_tfidf = vectorizer.transform(X_test)
# y_pred = model.predict(X_test_tfidf)
#
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))




#сдеалем легкий фласк
with open("news_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Модель и Vectorizer сохранены в 'news_classifier.pkl'")

app = Flask(__name__)


with open("news_classifier.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    X_tfidf = vectorizer.transform([text])
    pred = model.predict(X_tfidf)[0]

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    # Запуск на 5000 порту
    app.run(host="0.0.0.0", port=5000, debug=True)
