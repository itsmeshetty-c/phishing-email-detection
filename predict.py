"""
Phishing Email Prediction Tool
================================
Standalone script to perform inference on new email messages using the trained
Phishing Detection model pipeline.

Usage:
  1. Interactive CLI Mode:
     python predict.py

  2. Command-Line Argument Mode:
     python predict.py --text "Urgent: Account locked! Click http://192.168.1.1/verify" --subject "Security Alert"

Author: Internship Project Submission
"""

import sys
import os
import argparse
import joblib
import pandas as pd
from scipy.sparse import hstack

# Ensure phishing_detector is in python path to load PhishingFeatureExtractor class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phishing_detector import PhishingFeatureExtractor, URGENCY_KEYWORDS, SHORTENED_DOMAINS, SUSPICIOUS_TLD_PATTERNS


def load_pipeline(model_path=os.path.join("models", "full_pipeline.pkl")):
    """
    Loads the serialized model pipeline bundle.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found! Please run 'phishing_detector.py' first to train the model."
        )
    return joblib.load(model_path)


def analyze_risk_factors(email_text, subject, sender):
    """
    Extracts human-readable risk triggers for explainability.
    """
    triggers = []
    full_text = f"{subject} {email_text}".lower()

    # 1. Urgent keywords
    found_kw = [kw for kw in URGENCY_KEYWORDS if kw in full_text]
    if found_kw:
        triggers.append(f"Urgency phrases detected: {found_kw[:3]}")

    # 2. IP-based links
    import re
    if re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", email_text):
        triggers.append("Contains IP-address based raw URL link")

    # 3. Shortened URLs
    if any(short in email_text.lower() for short in SHORTENED_DOMAINS):
        triggers.append("Contains shortened URL link (e.g. bit.ly, tinyurl)")

    # 4. Suspicious domain keywords / TLDs
    found_tlds = [pat.replace(r"\b", "").replace("\\", "") for pat in SUSPICIOUS_TLD_PATTERNS if re.search(pat, email_text.lower() + " " + sender.lower())]
    if found_tlds:
        triggers.append(f"Suspicious domain / TLD patterns detected: {found_tlds[:3]}")

    # 5. Exclamation marks in subject
    if subject.count("!") >= 2:
        triggers.append(f"Multiple exclamation marks in subject ('{subject}')")

    if not triggers:
        triggers.append("No obvious heuristic phishing patterns detected.")

    return triggers


def predict_email(email_text, subject="", sender="", model_path=os.path.join("models", "full_pipeline.pkl")):
    """
    Predicts whether a given email is Phishing or Safe.

    Parameters:
        email_text (str): The email body text.
        subject (str): The subject line of the email.
        sender (str): The sender email address.
        model_path (str): Path to saved pipeline bundle.

    Returns:
        dict: Result containing label, confidence, probabilities, and risk factors.
    """
    bundle = load_pipeline(model_path)

    feature_extractor = bundle["feature_extractor"]
    scaler = bundle["scaler"]
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    # Format input into a dataframe structure
    df_input = pd.DataFrame([{
        "email_text": email_text,
        "subject": subject,
        "sender": sender
    }])

    combined_text = [f"{subject} {email_text}"]

    # Extract features
    X_num = feature_extractor.transform(df_input)
    X_num_scaled = scaler.transform(X_num)
    X_tfidf = vectorizer.transform(combined_text)

    X_combined = hstack([X_num_scaled, X_tfidf]).tocsr()

    # Predict probability and class
    probs = classifier.predict_proba(X_combined)[0]  # [prob_safe, prob_phishing]
    phishing_prob = probs[1]
    safe_prob = probs[0]

    label = "Phishing" if phishing_prob >= 0.50 else "Safe"
    confidence = phishing_prob if label == "Phishing" else safe_prob

    risk_triggers = analyze_risk_factors(email_text, subject, sender)

    return {
        "label": label,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.1f}%",
        "probabilities": {
            "Phishing": float(phishing_prob),
            "Safe": float(safe_prob)
        },
        "risk_factors": risk_triggers
    }


def run_cli_interactive():
    """
    Runs an interactive terminal loop for user testing.
    """
    print("=" * 65)
    print("      PHISHING EMAIL DETECTION - INTERACTIVE CLI TOOL      ")
    print("=" * 65)
    print("Type your email text below for instant classification.")
    print("Type 'exit' or 'quit' to close the program.\n")

    while True:
        try:
            print("-" * 65)
            subject = input("Enter Subject (optional, press Enter to skip): ").strip()
            sender = input("Enter Sender Email (optional, press Enter to skip): ").strip()
            print("Enter Email Body Text (press Enter then Ctrl+Z/D or type END on new line):")
            
            lines = []
            while True:
                line = input()
                if line.strip().upper() in ["END", "EXIT", "QUIT"]:
                    if line.strip().upper() in ["EXIT", "QUIT"] and not lines:
                        print("Exiting application...")
                        return
                    break
                lines.append(line)

            email_text = "\n".join(lines).strip()
            if not email_text and not subject:
                print("[WARNING] Please enter email text or subject to analyze.")
                continue

            result = predict_email(email_text, subject, sender)

            print("\n" + "=" * 45)
            print(" PREDICTION RESULT ")
            print("=" * 45)
            status_icon = "[ALERT] PHISHING DETECTED" if result['label'] == "Phishing" else "[SAFE] SAFE EMAIL"
            print(f"Classification : {status_icon}")
            print(f"Confidence     : {result['confidence_percentage']}")
            print(f"Phishing Prob  : {result['probabilities']['Phishing'] * 100:.1f}%")
            print(f"Safe Prob      : {result['probabilities']['Safe'] * 100:.1f}%")
            print("\nRisk Indicators Identified:")
            for trigger in result["risk_factors"]:
                print(f"  • {trigger}")
            print("=" * 45 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting application...")
            break


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Classify emails as Phishing or Safe using trained ML model.")
    parser.add_argument("-t", "--text", type=str, help="Email body text to evaluate.")
    parser.add_argument("-s", "--subject", type=str, default="", help="Email subject line.")
    parser.add_argument("-sender", "--sender", type=str, default="", help="Sender email address.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive CLI prompt.")

    args = parser.parse_args()

    if args.interactive or (not args.text and len(sys.argv) == 1):
        run_cli_interactive()
    elif args.text or args.subject:
        result = predict_email(args.text or "", args.subject, args.sender)
        print("\n" + "=" * 45)
        print(" PREDICTION RESULT ")
        print("=" * 45)
        status_icon = "[ALERT] PHISHING DETECTED" if result['label'] == "Phishing" else "[SAFE] SAFE EMAIL"
        print(f"Classification : {status_icon}")
        print(f"Confidence     : {result['confidence_percentage']}")
        print(f"Phishing Prob  : {result['probabilities']['Phishing'] * 100:.1f}%")
        print(f"Safe Prob      : {result['probabilities']['Safe'] * 100:.1f}%")
        print("\nRisk Indicators Identified:")
        for trigger in result["risk_factors"]:
            print(f"  • {trigger}")
        print("=" * 45 + "\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

