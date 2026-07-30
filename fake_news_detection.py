import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# STEP 1: Load Dataset
# ==============================

df = pd.read_csv("news.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

# ==============================
# STEP 2: Dataset Information
# ==============================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print(df.info())

# ==============================
# STEP 3: Missing Values
# ==============================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# ==============================
# STEP 4: Combine Title + Text
# ==============================

df["content"] = df["title"] + " " + df["text"]

print("\n" + "=" * 50)
print("COMBINED DATA")
print("=" * 50)
print(df[["content", "label"]].head())

# ==============================
# STEP 5: Features and Labels
# ==============================

X = df["content"]
y = df["label"]

# ==============================
# STEP 6: Split Dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==============================
# STEP 7: TF-IDF Vectorization
# ==============================

vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nTF-IDF Conversion Completed")
print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==============================
# STEP 8: Train Model
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ==============================
# STEP 9: Prediction
# ==============================

y_pred = model.predict(X_test)

print("\nPrediction Completed")

# ==============================
# STEP 10: Accuracy
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

# ==============================
# STEP 11: Classification Report
# ==============================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==============================
# STEP 12: Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# STEP 13: Test Your Own News
# ==============================

print("\n")
print("=" * 50)
print("FAKE NEWS DETECTOR")
print("=" * 50)

user_news = input("Enter a news headline or article:\n")

user_vector = vectorizer.transform([user_news])

prediction = model.predict(user_vector)

if prediction[0] == 1:
    print("\n✅ Result : REAL NEWS")
else:
    print("\n❌ Result : FAKE NEWS")