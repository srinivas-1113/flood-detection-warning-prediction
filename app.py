from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Temp", "Humidity", "Cloud Cover", "ANNUAL",
    "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec",
    "avgjune", "sub"
]

@app.route("/", methods=["GET", "POST"])
def home():
    probability = None
    warning = None

    if request.method == "POST":
        values = [
            float(request.form["Temp"]),
            float(request.form["Humidity"]),
            float(request.form["CloudCover"]),
            float(request.form["ANNUAL"]),
            float(request.form["JanFeb"]),
            float(request.form["MarMay"]),
            float(request.form["JunSep"]),
            float(request.form["OctDec"]),
            float(request.form["avgjune"]),
            float(request.form["sub"])
        ]

        input_df = pd.DataFrame([values], columns=feature_names)
        input_scaled = scaler.transform(input_df)

        probability = model.predict_proba(input_scaled)[0][1]

        if probability >= 0.8:
            warning = "🚨 SEVERE FLOOD WARNING"
        elif probability >= 0.6:
            warning = "⚠️ MODERATE FLOOD RISK"
        elif probability >= 0.4:
            warning = "🟡 LOW FLOOD RISK"
        else:
            warning = "🟢 NO FLOOD RISK"

    return render_template("index.html",
                           probability=probability,
                           warning=warning)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
