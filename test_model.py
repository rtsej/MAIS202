from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os
import joblib

app = Flask(__name__)
CORS(app)

MODEL_PATH = "saved_model.pkl"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"CSV read failed: {str(e)}"}), 400

    print(f"Training file columns: {df.columns.tolist()}")

    # Ensure required columns exist
    if 'Text' not in df.columns or 'Bias' not in df.columns:
        return jsonify({"error": "CSV must have 'Text' and 'Bias' columns."}), 400

    # Rename to internal expected names
    df = df.rename(columns={"Text": "text", "Bias": "label"})

    clf = make_pipeline(CountVectorizer(), LogisticRegression())
    clf.fit(df['text'], df['label'])

    try:
        joblib.dump(clf, MODEL_PATH)
        print("✅ Model saved successfully to", os.path.abspath(MODEL_PATH))
    except Exception as e:
        print("❌ Failed to save model:", e)
        return jsonify({"error": f"Failed to save model: {str(e)}"}), 500

    return jsonify({"message": "Model trained and saved successfully."})


@app.route('/data', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Please upload training data first."}), 400

    clf = joblib.load(MODEL_PATH)
    print("✅ Model loaded from disk.")

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"CSV read failed: {str(e)}"}), 400

    print(f"Prediction file columns: {df.columns.tolist()}")

    if 'Text' not in df.columns:
        return jsonify({"error": "CSV must have a 'Text' column."}), 400

    df = df.rename(columns={"Text": "text"})

    predictions = clf.predict(df['text'])
    output = [{"Text": text, "label": label} for text, label in zip(df['text'], predictions)]

    return jsonify({"data": output})

if __name__ == '__main__':
    app.run(debug=True)
