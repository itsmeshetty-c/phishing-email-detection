"""
Phishing Email Detection Pipeline
==================================
A production-grade machine learning model using Python and Scikit-Learn
to detect phishing emails based on URL indicators, urgency keywords, metadata,
and TF-IDF text features.

Author: Internship Project
License: MIT
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# 1. Custom Feature Extractor
# -----------------------------------------------------------------------------

# List of common urgency keywords and phishing trigger phrases
URGENCY_KEYWORDS = [
    "verify now", "account suspended", "click here", "limited time",
    "confirm password", "action required", "log in immediately",
    "unauthorized login", "billing issue", "tax refund", "claim prize",
    "win money", "urgent", "security alert", "password expires",
    "immediately", "final notice", "suspended", "unauthorized",
    "lock", "restricted", "unverified", "kyc", "compromised",
]

# Common URL shortener services
SHORTENED_DOMAINS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "adf.ly"
]

# Suspicious TLDs and typosquatting brand variations
SUSPICIOUS_TLD_PATTERNS = [
    r"\.xyz\b", r"\.top\b", r"\.club\b", r"\.info\b", r"\.tech\b",
    r"paypa1", r"amaz0n", r"g00gle", r"micr0soft", r"fedx", r"netflx",
    r"sec-", r"-verify", r"-update", r"-auth", r"-login", r"-security"
]


class PhishingFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer to extract domain-specific heuristic features
    from raw email text, subject, and sender columns.
    """

    def __init__(self):
        self.feature_names_ = [
            "url_count",
            "has_ip_url",
            "has_shortened_url",
            "max_url_length",
            "suspicious_domain_count",
            "urgency_keyword_count",
            "subject_exclamation_count",
            "subject_all_caps_count",
            "sender_suspicious_flag",
        ]

    def fit(self, X, y=None):
        return self

    def _extract_row_features(self, row):
        # Extract fields, handling missing values
        email_text = str(row.get("email_text", ""))
        subject = str(row.get("subject", ""))
        sender = str(row.get("sender", ""))
        full_text = f"{subject} {email_text}".lower()

        # 1. URL-based features
        urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", email_text, re.IGNORECASE)
        url_count = len(urls)

        # Presence of IP-based URL (e.g., http://192.168.1.1/login)
        has_ip_url = int(any(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) for url in urls))

        # Presence of shortened URL
        has_shortened_url = int(any(any(short in url.lower() for short in SHORTENED_DOMAINS) for url in urls))

        # Maximum length of URLs found
        max_url_length = max([len(u) for u in urls]) if urls else 0

        # Suspicious domain indicators in URLs
        suspicious_domain_count = 0
        for url in urls:
            url_lower = url.lower()
            for pattern in SUSPICIOUS_TLD_PATTERNS:
                if re.search(pattern, url_lower):
                    suspicious_domain_count += 1

        # 2. Keyword-based urgency features
        urgency_keyword_count = sum(1 for kw in URGENCY_KEYWORDS if kw in full_text)

        # 3. Subject metadata features
        subject_exclamation_count = subject.count("!")
        words = subject.split()
        subject_all_caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)

        # 4. Sender suspicious domain flag
        sender_lower = sender.lower()
        sender_suspicious_flag = int(any(re.search(pat, sender_lower) for pat in SUSPICIOUS_TLD_PATTERNS))

        return [
            url_count,
            has_ip_url,
            has_shortened_url,
            max_url_length,
            suspicious_domain_count,
            urgency_keyword_count,
            subject_exclamation_count,
            subject_all_caps_count,
            sender_suspicious_flag,
        ]

    def transform(self, X):
        """
        X can be a pandas DataFrame with columns ['email_text', 'subject', 'sender']
        or a list of dicts.
        """
        if isinstance(X, pd.DataFrame):
            features = [self._extract_row_features(row) for _, row in X.iterrows()]
        elif isinstance(X, list):
            features = [self._extract_row_features(item) for item in X]
        else:
            raise ValueError("Input X must be a pandas DataFrame or list of dicts.")

        return np.array(features, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


# -----------------------------------------------------------------------------
# 2. Model Training & Evaluation Pipeline
# -----------------------------------------------------------------------------

def load_and_preprocess_data(data_path="dataset.csv"):
    """
    Loads dataset from CSV and prepares text & label arrays.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df.fillna("", inplace=True)

    print(f"[INFO] Loaded dataset with {len(df)} records.")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts()}\n")

    # Combine subject and body for TF-IDF vectorization
    combined_text = df["subject"] + " " + df["email_text"]

    return df, combined_text, df["label"]


def train_and_evaluate():
    """
    Main training workflow:
    1. Feature extraction & TF-IDF vectorization
    2. Train/Test split (80/20 stratified)
    3. Model training & comparison (Logistic Regression vs. Random Forest)
    4. Visualization generation
    5. Artifact saving
    """
    # Create directories for outputs and models if they don't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    df, combined_text, labels = load_and_preprocess_data("dataset.csv")

    # Encode target labels: Phishing = 1, Safe = 0
    y = np.where(labels == "Phishing", 1, 0)

    # 2. Train/Test Split (80/20 Stratified)
    df_train, df_test, text_train, text_test, y_train, y_test = train_test_split(
        df, combined_text, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"[INFO] Training set size: {len(df_train)} | Test set size: {len(df_test)}")

    # 3. Feature Extraction
    print("[INFO] Extracting domain features and TF-IDF text features...")
    feature_extractor = PhishingFeatureExtractor()
    scaler = StandardScaler()

    # Domain features
    X_num_train = feature_extractor.fit_transform(df_train)
    X_num_test = feature_extractor.transform(df_test)

    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=300,
        stop_words="english",
        sublinear_tf=True
    )
    X_tfidf_train = vectorizer.fit_transform(text_train)
    X_tfidf_test = vectorizer.transform(text_test)

    # Combine numerical features and TF-IDF features
    X_train_combined = hstack([X_num_train_scaled, X_tfidf_train]).tocsr()
    X_test_combined = hstack([X_num_test_scaled, X_tfidf_test]).tocsr()

    # Feature names
    num_feature_names = feature_extractor.feature_names_
    tfidf_feature_names = list(vectorizer.get_feature_names_out())
    all_feature_names = num_feature_names + tfidf_feature_names

    # 4. Model Training & Comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    print("\n" + "=" * 60)
    print(" MODEL PERFORMANCE EVALUATION MATRIX ")
    print("=" * 60)

    for name, clf in models.items():
        clf.fit(X_train_combined, y_train)
        y_pred = clf.predict(X_test_combined)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "model": clf,
            "y_pred": y_pred,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        print(f"\n---> {name}:")
        print(f"     Accuracy : {acc * 100:.2f}%")
        print(f"     Precision: {prec * 100:.2f}%")
        print(f"     Recall   : {rec * 100:.2f}%")
        print(f"     F1-Score : {f1 * 100:.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf

    print("\n" + "=" * 60)
    print(f" BEST PERFORMING MODEL: {best_model_name} (F1-Score: {best_f1 * 100:.2f}%)")
    print("=" * 60 + "\n")

    # Print Classification Report for Best Model
    y_best_pred = results[best_model_name]["y_pred"]
    print(f"[INFO] Detailed Classification Report ({best_model_name}):\n")
    print(classification_report(y_test, y_best_pred, target_names=["Safe", "Phishing"]))

    # 5. Visualizations
    # 5a. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix ({best_model_name})", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join("outputs", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Confusion Matrix plot to '{cm_path}'.")

    # 5b. Feature Importance Plot (Random Forest)
    rf_clf = results["Random Forest"]["model"]
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features

    top_features = [all_feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(9, 5.5))
    colors = ["#1f77b4" if f in num_feature_names else "#ff7f0e" for f in top_features]
    plt.barh(range(len(top_features)), top_importances[::-1], color=colors[::-1], align="center")
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=10)
    plt.xlabel("Gini Feature Importance", fontsize=12)
    plt.title("Top 15 Most Discriminative Features (Random Forest)", fontsize=14, pad=12)
    
    # Custom legend for feature categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Heuristic / URL / Domain Feature"),
        Patch(facecolor="#ff7f0e", label="TF-IDF Text Feature")
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    fi_path = os.path.join("outputs", "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Feature Importance plot to '{fi_path}'.")

    # 6. Save Model Artifacts
    pipeline_bundle = {
        "feature_extractor": feature_extractor,
        "scaler": scaler,
        "vectorizer": vectorizer,
        "classifier": best_model_obj,
        "best_model_name": best_model_name,
        "all_feature_names": all_feature_names,
        "results": {
            k: {m: v for m, v in val.items() if m != "model" and m != "y_pred"}
            for k, val in results.items()
        }
    }

    joblib.dump(pipeline_bundle, os.path.join("models", "full_pipeline.pkl"))
    joblib.dump(best_model_obj, os.path.join("models", "model.pkl"))
    joblib.dump(vectorizer, os.path.join("models", "vectorizer.pkl"))

    print(f"[SUCCESS] Serialized trained model bundle to 'models/full_pipeline.pkl'.")
    print(f"[SUCCESS] Pipeline training complete!\n")


if __name__ == "__main__":
    train_and_evaluate()
