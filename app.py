import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------

df = pd.read_csv("fake_news_clean.csv")

# -----------------------------
# 2. Prepare data
# -----------------------------

# Combine title and text (better results, still simple)
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# -----------------------------
# 3. Train / test split
# -----------------------------

# Without stratification, initial version before improvement:
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42
#)

# With stratification:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Text → numbers
# -----------------------------
# Initial simple version:
#vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Improved version with unigrams + bigrams:
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. Train model
# -----------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 6. Evaluate
# -----------------------------

y_pred = model.predict(X_test_vec)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDetailed report:")
print(classification_report(y_test, y_pred, digits=4))
