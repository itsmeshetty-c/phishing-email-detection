"""
Phishing Email Detection - Desktop Web Application Backend
============================================================
A Flask-powered web server providing an interactive desktop GUI for real-time
phishing email scanning, threat intelligence, and model visualization.

Usage:
    python app.py
"""

import os
import sys
import webbrowser
import threading
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure predict module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_email, load_pipeline
from phishing_detector import PhishingFeatureExtractor

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global pipeline cache
PIPELINE_BUNDLE = None


def get_pipeline():
    global PIPELINE_BUNDLE
    if PIPELINE_BUNDLE is None:
        model_path = os.path.join("models", "full_pipeline.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file missing! Please run 'phishing_detector.py' first.")
        PIPELINE_BUNDLE = load_pipeline(model_path)
    return PIPELINE_BUNDLE


# Preset sample emails for 1-click testing
SAMPLE_EMAILS = [
    {
        "id": "phishing_1",
        "title": "🚨 Bank Account Locked",
        "type": "Phishing",
        "subject": "URGENT: Chase Account Locked",
        "sender": "alerts@chase-security-update.xyz",
        "email_text": "Urgent: Your Chase bank account has been locked due to suspicious activity. Please click http://192.168.1.105/chase/verify to verify your identity immediately and restore access."
    },
    {
        "id": "phishing_2",
        "title": "🚨 PayPal Password Alert",
        "type": "Phishing",
        "subject": "Security Alert: Unauthorized Login",
        "sender": "service@paypa1-support.com",
        "email_text": "Dear customer, your PayPal account was accessed from an unknown device in Moscow. Click http://bit.ly/paypal-sec-auth to confirm password and secure your account now."
    },
    {
        "id": "safe_1",
        "title": "✅ Corporate Team Sync",
        "type": "Safe",
        "subject": "Weekly Team Sync - Agenda & Slides",
        "sender": "alex.smith@company.com",
        "email_text": "Hi team, attached is the agenda for our weekly project sync tomorrow at 10 AM. Please review the slides before the meeting."
    },
    {
        "id": "safe_2",
        "title": "✅ Amazon Order Receipt",
        "type": "Safe",
        "subject": "Order Confirmation #49281",
        "sender": "orders@amazon.com",
        "email_text": "Thank you for your order #49281. We have received your payment and your items are being prepared for shipment. Tracking details will follow."
    }
]


@app.route("/")
def index():
    """Renders the desktop dashboard GUI."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts email content, performs ML inference, and returns prediction details
    along with extracted numerical feature values.
    """
    try:
        data = request.get_json() or {}
        email_text = data.get("email_text", "").strip()
        subject = data.get("subject", "").strip()
        sender = data.get("sender", "").strip()

        if not email_text and not subject:
            return jsonify({"error": "Please provide email text or subject line to analyze."}), 400

        # Run prediction
        result = predict_email(email_text, subject, sender)

        # Extract raw feature values for detailed inspection
        bundle = get_pipeline()
        extractor = bundle["feature_extractor"]
        df_input = pd.DataFrame([{"email_text": email_text, "subject": subject, "sender": sender}])
        raw_num_features = extractor.transform(df_input)[0]

        feature_dict = dict(zip(extractor.feature_names_, raw_num_features.tolist()))

        result["extracted_features"] = feature_dict
        result["best_model_name"] = bundle.get("best_model_name", "Logistic Regression")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns sample email presets."""
    return jsonify(SAMPLE_EMAILS)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns model training benchmark results."""
    try:
        bundle = get_pipeline()
        results = bundle.get("results", {})
        return jsonify({
            "best_model_name": bundle.get("best_model_name", "Logistic Regression"),
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/outputs/<filename>")
def serve_output_file(filename):
    """Serves output image files (confusion matrix, feature importance)."""
    return send_from_directory("outputs", filename)


def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Ensure model exists before launching server
    try:
        get_pipeline()
    except FileNotFoundError:
        print("[WARNING] Trained model not found. Running training pipeline first...")
        from phishing_detector import train_and_evaluate
        train_and_evaluate()

    print("\n" + "=" * 60)
    print(" [LAUNCH] STARTING PHISHING DETECTION DESKTOP WEB APP ")
    print("=" * 60)
    print(" Access Dashboard: http://127.0.0.1:5000/")
    print(" Press Ctrl+C in terminal to stop server.\n")

    # Open browser after 1.2 seconds delay
    threading.Timer(1.2, open_browser).start()

    app.run(host="127.0.0.1", port=5000, debug=False)

