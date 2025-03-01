import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("labelled_newscatcher_dataset.csv", sep=";")


print("Первые строки датасета:")
print(df.head())
print("\nКолонки:", df.columns)


TEXT_COL = "title"
LABEL_COL = "topic"


df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)

print("Форма датафрейма после очистки:", df.shape)
print("Пример меток:")
print(df[LABEL_COL].value_counts())

X = df[TEXT_COL].values
y = df[LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF + LOGISTIC REGRESSION
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. ОЦЕНКА НА ТЕСТЕ
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Немного не поняла как парсить с ссылки и сделала тоже самое
