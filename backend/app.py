from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, re, string

app = Flask(__name__)
CORS(app)

# Load model + TF-IDF
model = pickle.load(open("model/ticket_classifier.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

@app.route("/")
def home():
    return "✅ Customer Support Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    return jsonify({"department": pred})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
