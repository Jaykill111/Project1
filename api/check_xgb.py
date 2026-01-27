try:
    import xgboost
    print(f"XGBoost version: {xgboost.__version__}")
except ImportError:
    print("XGBoost NOT installed")
except Exception as e:
    print(f"Error importing XGBoost: {e}")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("Scikit-learn NOT installed")
