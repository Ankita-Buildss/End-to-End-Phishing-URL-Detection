import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict, Any, Optional

from src.config_loader import load_config
from src.website_feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)

# =========================================================
# PATHS + LOAD ARTIFACTS (config-driven)
# =========================================================

CONFIG = load_config()
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR  = PROJECT_ROOT / CONFIG["artifacts"]["directory"]
SCALER_PATH    = ARTIFACTS_DIR / CONFIG["artifacts"]["scaler_filename"]
XGB_MODEL_PATH = ARTIFACTS_DIR / CONFIG["artifacts"]["xgb_model_filename"]
ANN_MODEL_PATH = ARTIFACTS_DIR / CONFIG["artifacts"]["ann_model_filename"]


def _safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Fix: Run training pipeline first so artifacts get created."
        )
    return joblib.load(path)


SCALER    = _safe_load(SCALER_PATH)
XGB_MODEL = _safe_load(XGB_MODEL_PATH)
ANN_MODEL = _safe_load(ANN_MODEL_PATH)

# =========================================================
# THRESHOLDS
# =========================================================
# proba[1] = P(legit).
# pred = 1 (legit)    if proba[1] >= threshold
# pred = 0 (phishing) otherwise
#
# Lower  → easier to be called legit (fewer false positives on legit sites)
# Higher → harder to be called legit (more conservative / catches more phishing)
#
# XGBoost and ANN are calibrated separately because they produce different
# probability scales — ANN tends to be more confident, XGBoost more spread out.
XGB_LEGIT_THRESHOLD = 0.30   # XGBoost: lower = fewer false positives
ANN_LEGIT_THRESHOLD = 0.35   # ANN: slightly higher since it's already generous

# =========================================================
# HELPERS
# =========================================================

def decode_label(pred: int) -> str:
    # Dataset convention: 1 = legit, 0 = phishing
    return "Legit Website" if int(pred) == 1 else "Phishing Website"


def _get_expected_columns_from_scaler() -> Optional[list]:
    if hasattr(SCALER, "feature_names_in_"):
        return list(SCALER.feature_names_in_)
    return None


EXPECTED_COLUMNS = _get_expected_columns_from_scaler()


def validate_and_build_df(input_dict: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a Python dict of feature_name -> value.")

    df = pd.DataFrame([input_dict])

    if EXPECTED_COLUMNS is not None:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}\n"
                f"Expected columns (total {len(EXPECTED_COLUMNS)}): {EXPECTED_COLUMNS}"
            )
        if extra:
            df = df.drop(columns=extra)
        df = df[EXPECTED_COLUMNS]

    return df


# =========================================================
# WHOIS CACHE  (avoids 2x network calls for same domain)
# =========================================================

@lru_cache(maxsize=256)
def _whois_cached(domain: str):
    import whois
    return whois.whois(domain)


# =========================================================
# SUBDOMAIN SCORING  (replaces blunt having_Sub_Domain flag)
# =========================================================

def subdomain_score(domain: str) -> int:
    """
    Smarter subdomain check.

    The old version flagged ANY subdomain as suspicious:
        chat.openai.com              → 1  (WRONG — totally legit)
        mail.google.com              → 1  (WRONG)
        login.secure.verify.xyz      → 1  (correct, but same score as above)

    New logic:
        Strip www. first (not a real subdomain).
        0 or 1 dot remaining → 0  (e.g. openai.com or chat.openai.com = fine)
        2+ dots remaining    → 1  (e.g. login.verify.paypal.com = suspicious)

    This keeps chat.openai.com, mail.google.com, docs.github.com as 0
    while still catching deeply nested phishing subdomains.
    """
    if domain.startswith("www."):
        domain = domain[4:]
    return 1 if domain.count(".") >= 2 else 0


# =========================================================
# FEATURE PATCHING
# =========================================================

def patch_features(input_dict: dict, url: str) -> dict:
    """
    Patch features that FeatureExtractor gets wrong for legit sites.

    having_Sub_Domain: replaced with smarter subdomain_score so that
    chat.openai.com is not penalised for having one clean subdomain.

    _raw_domain is stored internally for rule_based_check and stripped
    before features are sent to the model (underscore prefix = internal key).
    """
    from urllib.parse import urlparse
    domain = urlparse(url).hostname or ""

    patched = dict(input_dict)
    patched["having_Sub_Domain"] = subdomain_score(domain)
    patched["_raw_domain"] = domain
    return patched


# =========================================================
# RULE-BASED OVERRIDE
# =========================================================

def rule_based_check(input_dict: dict) -> bool:
    """
    Hard rules that override the ML model.
    Returns True if URL is definitely phishing.

    Intentionally strict — a legit site like google.com or chat.openai.com
    must never trigger this.
    """
    # IP address directly in URL → always phishing
    if input_dict.get("having_IP_Address") == 1:
        return True

    no_ssl  = input_dict.get("SSLfinal_State") == 0
    no_dns  = input_dict.get("DNSRecord") == 0
    young   = input_dict.get("age_of_domain") == 0
    new_reg = input_dict.get("Domain_registeration_length") == 0

    # All four must be true simultaneously — avoids false positives
    if no_dns and no_ssl and young and new_reg:
        return True

    # 6+ signals required (raised from 5 to reduce false positives)
    domain = input_dict.get("_raw_domain", "")
    suspicious_count = sum([
        input_dict.get("Prefix_Suffix", 0),
        subdomain_score(domain),           # smarter score, not raw flag
        input_dict.get("Abnormal_URL", 0),
        1 if no_dns  else 0,
        1 if no_ssl  else 0,
        1 if young   else 0,
        1 if new_reg else 0,
    ])
    if suspicious_count >= 6:
        return True

    return False


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_input(input_dict: Dict[str, Any]) -> np.ndarray:
    df = validate_and_build_df(input_dict)
    X_scaled = SCALER.transform(df)
    return X_scaled


# =========================================================
# PREDICT
# =========================================================

def predict(
    input_dict: Dict[str, Any],
    model_type: str = "xgboost",
    url: str = ""
) -> Dict[str, Any]:
    """
    Predict phishing vs legit.

    Label convention (matches training data):
        1 = Legit Website
        0 = Phishing Website

    predict_proba output:
        proba[0] = P(phishing)
        proba[1] = P(legit)

    Always pass `url` so smarter subdomain patching is applied.
    """

    # --- Patch features (smarter subdomain logic) ---
    if url:
        input_dict = patch_features(input_dict, url)

    # --- Rule-based override FIRST ---
    if rule_based_check(input_dict):
        return {
            "model":                model_type,
            "override":             True,
            "prediction":           0,
            "result_text":          "Phishing Website",
            "phishing_probability": 1.0,
            "legit_probability":    0.0,
        }

    # Strip internal keys before sending to model
    model_input = {k: v for k, v in input_dict.items() if not k.startswith("_")}

    # --- Select model + threshold ---
    if model_type.lower() == "xgboost":
        model     = XGB_MODEL
        threshold = XGB_LEGIT_THRESHOLD
    elif model_type.lower() in ["ann", "mlp", "mlpclassifier"]:
        model      = ANN_MODEL
        model_type = "ann"
        threshold  = ANN_LEGIT_THRESHOLD
    else:
        raise ValueError("model_type must be 'xgboost' or 'ann'.")

    # --- Preprocess ---
    X_scaled = preprocess_input(model_input)

    # --- Predict (single predict_proba call, result reused) ---
    phishing_prob = None
    legit_prob    = None

    if hasattr(model, "predict_proba"):
        proba         = model.predict_proba(X_scaled)[0]
        phishing_prob = float(proba[0])
        legit_prob    = float(proba[1])
        pred          = 1 if legit_prob >= threshold else 0
    else:
        pred = int(model.predict(X_scaled)[0])

    return {
        "model":                model_type,
        "override":             False,
        "prediction":           pred,
        "result_text":          decode_label(pred),
        "phishing_probability": phishing_prob,
        "legit_probability":    legit_prob,
    }


# =========================================================
# CLI DEMO
# =========================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = FeatureExtractor()

    urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://chat.openai.com",
        "https://mail.google.com",
        "https://docs.github.com",
        "http://secure-paypal-login.verify-account.xyz",
        "http://192.168.1.1/login",
    ]

    for url in urls:
        from urllib.parse import urlparse
        domain = urlparse(url).hostname or ""
        features = extractor.extract(url)

        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"  Raw subdomain flag : {features.get('having_Sub_Domain')}")
        print(f"  Patched subdomain  : {subdomain_score(domain)}")
        print("  XGBoost:", predict(features, model_type="xgboost", url=url))
        print("  ANN    :", predict(features, model_type="ann",     url=url))