import pandas as pd
import re, string, pickle, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- SETTINGS ---------------- #
INPUT_FILE = "../dataset/customer_support_tickets.csv"   # your large 3M dataset
SAMPLE_SIZE = 10000                                   # total 10k sample
TRAIN_RATIO = 0.9                                     # 9000 train, 1000 test
MODEL_PATH = "model/ticket_classifier.pkl"
VEC_PATH = "model/tfidf.pkl"
OUTPUT_PREDICTION_FILE = "../dataset/testing_output.csv"
RANDOM_STATE = 42
# ------------------------------------------ #

print("📥 Loading large dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"✅ Original dataset rows: {len(df)}")

# Random sample of 10,000 rows
df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"✅ Sampled {SAMPLE_SIZE} rows from dataset")

# Use only 'text' column
if 'text' not in df.columns:
    raise ValueError("❌ 'text' column not found in dataset!")

# Create synthetic labels (since original dataset unlabeled)
def assign_label(txt):
    txt = str(txt).lower()
    if any(k in txt for k in ["payment", "charge", "billing", "refund"]):
        return "Billing"
    elif any(k in txt for k in ["error", "not working", "issue", "problem", "fix"]):
        return "Technical"
    elif any(k in txt for k in ["account", "login", "password", "user"]):
        return "Account"
    else:
        return "General"

df["label"] = df["text"].apply(assign_label)

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# Split into 90% train, 10% test
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE, stratify=y
)

print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully! Accuracy: {acc:.3f}\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(tfidf, open(VEC_PATH, "wb"))
print(f"🎯 Model + TFIDF saved to {MODEL_PATH} and {VEC_PATH}")

# Save test predictions for presentation
pred_df = pd.DataFrame({
    "Original_Text": X_test,
    "True_Label": y_test,
    "Predicted_Label": y_pred
})
pred_df.to_csv(OUTPUT_PREDICTION_FILE, index=False)
print(f"📊 Predictions saved to {OUTPUT_PREDICTION_FILE}")
print(f"📅 Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# -------------------------------------------------------------
# 📊 CONFUSION MATRIX VISUALIZATION
# -------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(7, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Customer Support Ticket Classifier", fontsize=14)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../dataset/confusion_matrix.png")
plt.close()
print("✅ Confusion Matrix saved as 'confusion_matrix.png'")

# -------------------------------------------------------------
# 📈 CLASS-WISE PRECISION, RECALL, F1-SCORE CHART
# -------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=model.classes_)

scores_df = pd.DataFrame({
    "Label": model.classes_,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1
})

plt.figure(figsize=(8, 5))
scores_df.plot(x="Label", kind="bar", rot=0)
plt.title("Precision, Recall, F1-Score per Class", fontsize=13)
plt.xlabel("Ticket Type")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("../dataset/class_scores.png")
plt.close()
print("✅ Class-wise performance chart saved as 'class_scores.png'")