print("Second tasks!")

# Topic Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


data = pd.read_csv("labelled_newscatcher_dataset.csv", sep=";")

text_column = "title"
label_column = "topic"

data.dropna(subset=[text_column, label_column], inplace=True)
print("Размер набора после удаления пустыз значений:", data.shape)
print("Распределение по категориям:\n", data[label_column].value_counts())


# Тренируем и тестируем
X = data[text_column].values
y = data[label_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# pipeline сборка контейнера
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),   # Можно указать стоп-слова для нужного языка
    ("clf", LogisticRegression(max_iter=1000))
])


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

