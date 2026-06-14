from flask import Flask, render_template, request
import joblib
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

model = joblib.load("piracy_detector.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk_score = None
    extracted_text = ""

    if request.method == "POST":
        url = request.form["url"]

        try:
            response = requests.get(url, timeout=10)

            soup = BeautifulSoup(response.text, "html.parser")

            extracted_text = soup.get_text(separator=" ", strip=True)

            result = model.predict([extracted_text])[0]
            probability = model.predict_proba([extracted_text])[0]

            if result == 1:
                prediction = "⚠️ Potential Piracy Content"
                risk_score = round(probability[1] * 100, 2)
            else:
                prediction = "✅ Legitimate Content"
                risk_score = round(probability[0] * 100, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        risk_score=risk_score,
        extracted_text=extracted_text[:500]
    )

if __name__ == "__main__":
    app.run(debug=True)