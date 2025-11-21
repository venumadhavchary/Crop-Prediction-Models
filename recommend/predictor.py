# recommend/predictor.py
import os, traceback
from django.conf import settings
import numpy as np

ML_DIR = getattr(settings, 'ML_MODELS_DIR', os.path.join(settings.BASE_DIR, 'ml_models'))

# Files and order to try
MODEL_CANDIDATES = [
    'rf_model.pkl',
    'xgb_model.pkl',
    'lgbm_model.pkl',
    'cat_model.pkl',
    'crop_model.joblib',
    'crop_model.pkl',
    'crop_model.h5',
]

SCALER_FILE = os.path.join(ML_DIR, 'scaler.pkl')
LABEL_ENCODER_FILE = os.path.join(ML_DIR, 'label_encoder.pkl')

# Default feature order: CHANGE this if your training used different order.
FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


_MODEL = None
_MODEL_KIND = None  # 'sklearn'|'xgb'|'lgbm'|'catboost'|'keras'

def find_model_path():
    for fname in MODEL_CANDIDATES:
        path = os.path.join(ML_DIR, fname)
        if os.path.exists(path):
            return path
    # fallback: any file in folder
    for f in os.listdir(ML_DIR):
        p = os.path.join(ML_DIR, f)
        if os.path.isfile(p):
            return p
    return None

def load_obj(path):
    """Helper: try joblib then pickle (latin1 fallback)."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass
    try:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        try:
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception:
            raise

def load_scaler():
    if os.path.exists(SCALER_FILE):
        try:
            return load_obj(SCALER_FILE)
        except Exception:
            return None
    return None

def load_label_encoder():
    if os.path.exists(LABEL_ENCODER_FILE):
        try:
            return load_obj(LABEL_ENCODER_FILE)
        except Exception:
            return None
    return None

def load_model():
    global _MODEL, _MODEL_KIND
    if _MODEL is not None:
        return _MODEL
    model_path = find_model_path()
    if not model_path:
        raise FileNotFoundError(f"No model file found in {ML_DIR}")
    # 1) try load_obj (joblib/pickle)
    try:
        m = load_obj(model_path)
        _MODEL = m
        # Heuristic: inspect module/class names to guess kind
        name = type(m).__module__.lower()
        if 'xgboost' in name:
            _MODEL_KIND = 'xgb'
        elif 'lightgbm' in name or 'lgbm' in name:
            _MODEL_KIND = 'lgbm'
        elif 'catboost' in name:
            _MODEL_KIND = 'catboost'
        else:
            _MODEL_KIND = 'sklearn'
        return _MODEL
    except Exception:
        pass
    # 2) try CatBoost native loader
    try:
        from catboost import CatBoost
        cb = CatBoost()
        cb.load_model(model_path)
        _MODEL = cb
        _MODEL_KIND = 'catboost'
        return _MODEL
    except Exception:
        pass
    # 3) try keras .h5
    try:
        from tensorflow.keras.models import load_model as keras_load
        _MODEL = keras_load(model_path)
        _MODEL_KIND = 'keras'
        return _MODEL
    except Exception:
        pass
    raise RuntimeError(f"Failed to load model file: {model_path}\n{traceback.format_exc()}")

def preprocess_input(features: dict):
    X = [features.get(k, 0) for k in FEATURE_ORDER]
    arr = np.array([X], dtype=float)
    scaler = load_scaler()
    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except Exception:
            # ignore scaler errors and use raw arr
            pass
    return arr

def decode_label(raw_label):
    le = load_label_encoder()
    if le is None:
        return str(raw_label)
    try:
        # if the label encoder expects numeric index
        try:
            return str(le.inverse_transform([raw_label])[0])
        except Exception:
            # else maybe raw_label is index-like
            return str(le.classes_[int(raw_label)])
    except Exception:
        # final fallback
        try:
            return str(le.inverse_transform([raw_label])[0])
        except Exception:
            return str(raw_label)

def predict(features: dict):
    """
    features: dict with keys matching FEATURE_ORDER (season_encoded included)
    returns dict: {'top_crop':..., 'scores': {...}} or {'prediction': ...} or {'error': ...}
    """
    try:
        model = load_model()
    except Exception as e:
        return {'error': str(e)}

    X = preprocess_input(features)

    # catboost
    if _MODEL_KIND == 'catboost':
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None
        if probs is not None and hasattr(probs, 'ndim') and probs.ndim == 2:
            probs = probs[0].tolist()
            le = load_label_encoder()
            if le is not None and hasattr(le, 'classes_'):
                classes = [str(c) for c in le.classes_]
            else:
                classes = [f'class_{i}' for i in range(len(probs))]
            scores = {c: float(round(float(p), 6)) for c, p in zip(classes, probs)}
            top = classes[int(np.argmax(probs))]
            return {'top_crop': top, 'scores': scores}
        # fallback predict
        try:
            p = model.predict(X)
            return {'prediction': str(p[0])}
        except Exception as e:
            return {'error': str(e)}

    # sklearn-like
    try:
        if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
            probs = model.predict_proba(X)[0]
            classes = [str(c) for c in model.classes_]
            scores = {c: float(round(float(p), 6)) for c, p in zip(classes, probs)}
            top_raw = classes[int(np.argmax(probs))]
            # decode label if possible
            try:
                top = decode_label(top_raw)
            except Exception:
                top = top_raw
            return {'top_crop': top, 'scores': scores}
    except Exception:
        pass

    # keras
    if _MODEL_KIND == 'keras':
        try:
            preds = model.predict(X)
            if preds.ndim == 2 and preds.shape[1] > 1:
                probs = preds[0].tolist()
                classes = [f'class_{i}' for i in range(len(probs))]
                scores = {c: float(round(p, 6)) for c, p in zip(classes, probs)}
                top = classes[int(np.argmax(probs))]
                return {'top_crop': top, 'scores': scores}
            return {'prediction': float(preds.ravel()[0])}
        except Exception as e:
            return {'error': str(e)}

    # fallback predict
    try:
        pred = model.predict(X)
        if hasattr(pred, '__iter__'):
            val = pred[0]
            try:
                return {'top_crop': decode_label(val)}
            except Exception:
                return {'prediction': float(val) if isinstance(val, (int, float)) else str(val)}
        return {'prediction': float(pred)}
    except Exception as e:
        return {'error': str(e)}
