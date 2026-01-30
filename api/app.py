"""
EPL SCOPE - Flask REST API
Premier League Corner Prediction API
"""
import sys
import os
import logging
from functools import wraps
from datetime import timedelta
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add api directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import glob
import requests
from dotenv import load_dotenv
from features import FEATURES, compute_features_for_match, compute_features_for_goals_match, get_team_statistics, get_head_to_head

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================
def handle_errors(f):
    """Decorator to handle errors consistently across all endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"ValueError in {f.__name__}: {e}")
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400
        except KeyError as e:
            logger.error(f"KeyError in {f.__name__}: {e}")
            return jsonify({"error": f"Missing parameter: {str(e)}"}), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}", exc_info=True)
            return jsonify({"error": "Internal server error", "details": str(e) if os.environ.get('FLASK_ENV') == 'development' else None}), 500
    return decorated_function

# ============================================================================
# DATA CACHE - Reduce CSV downloads
# ============================================================================
class DataCache:
    """In-memory cache for CSV data with TTL (Time To Live)."""
    def __init__(self, ttl_seconds=300):  # Cache for 5 minutes
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        """Get cached data if not expired."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache HIT for {key}")
                return data
            else:
                del self.cache[key]
                logger.info(f"Cache EXPIRED for {key}")
        return None
    
    def set(self, key, value):
        """Cache data with current timestamp."""
        self.cache[key] = (value, time.time())
        logger.info(f"Cache SET for {key}")
    
    def clear(self, key=None):
        """Clear specific key or all cache."""
        if key:
            self.cache.pop(key, None)
        else:
            self.cache.clear()

data_cache = DataCache(ttl_seconds=300)  # 5 minute cache

# ============================================================================
# INPUT VALIDATION
# ============================================================================
def validate_team(team_name):
    """Validate team name input."""
    if not team_name:
        raise ValueError("Team name cannot be empty")
    if not isinstance(team_name, str):
        raise ValueError("Team name must be a string")
    if len(team_name) > 100:
        raise ValueError("Team name too long (max 100 chars)")
    return team_name.strip()

def validate_threshold(threshold):
    """Validate threshold input."""
    try:
        th = float(threshold)
        if th <= 0:
            raise ValueError("Threshold must be positive")
        if th > 20:
            raise ValueError("Threshold too high (max 20)")
        return th
    except (TypeError, ValueError):
        raise ValueError(f"Invalid threshold: {threshold}")

def validate_league(league_code):
    """Validate league code."""
    if league_code not in LEAGUES:
        raise ValueError(f"Invalid league code: {league_code}. Valid codes: {list(LEAGUES.keys())}")
    return league_code



# Add cache control decorator
def add_cache_header(max_age=300):
    """Add cache control headers to response (default 5 min)."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = jsonify(f(*args, **kwargs)) if not isinstance(f(*args, **kwargs), dict) else f(*args, **kwargs)
            if isinstance(response, dict):
                response = jsonify(response)
            response.cache_control.max_age = max_age
            response.cache_control.public = True
            return response
        return decorated_function
    return decorator

CORS(app, resources={r"/api/*": {
    "origins": [
        "https://corner.qnguyen3.dev",
        "https://project1-six-flame.vercel.app",
        "https://myproject-woad-theta.vercel.app",
        "https://*.vercel.app",  # Accept any Vercel deployment
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8080"
    ],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

# Configuration - Multi-League Support
LEAGUES = {
    'E0': {'name': 'Premier League', 'country': 'England', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'iso': 'gb-eng'},
    'SP1': {'name': 'La Liga', 'country': 'Spain', 'flag': '🇪🇸', 'iso': 'es'},
    'I1': {'name': 'Serie A', 'country': 'Italy', 'flag': '🇮🇹', 'iso': 'it'},
    'D1': {'name': 'Bundesliga', 'country': 'Germany', 'flag': '🇩🇪', 'iso': 'de'},
    'F1': {'name': 'Ligue 1', 'country': 'France', 'flag': '🇫🇷', 'iso': 'fr'},
}

# Current selected league (default: Premier League)
CURRENT_LEAGUE = 'E0'

# Dynamic URLs based on selected league
def get_data_url(league_code: str, season: str = '2526') -> str:
    """Generate data URL for given league and season."""
    return f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'

DATA_URL = get_data_url(CURRENT_LEAGUE, '2526')
HISTORICAL_SEASONS = ['2324', '2425', '2526']  # Last 3 seasons for H2H
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
# Primary and fallback LLM models (configurable via .env)
# Google Gemini support removed - using OpenRouter only
# LLM_MODEL = os.environ.get('LLM_MODEL', 'google/gemini-2.0-flash-exp:free')
# LLM_FALLBACK_MODELS = [
#     m.strip() for m in os.environ.get('LLM_FALLBACK_MODELS', '').split(',') if m.strip()
# ]

# Model configurations - CORNERS (threshold -> window, confidence)
# Updated with optimized thresholds from analysis (2026-01-07)
MODEL_CONFIGS = {
    8.5: {'window': 5, 'confidence': 0.70},
    9.5: {'window': 5, 'confidence': 0.60},
    10.5: {'window': 5, 'confidence': 0.65},
    11.5: {'window': 5, 'confidence': 0.70},
    12.5: {'window': 5, 'confidence': 0.70},
}

# Model configurations - GOALS (threshold -> window, confidence)
# Updated with external data features (2026-01-27)
# V3 baseline, now using external model with +14% AUC for 2.5
GOALS_MODEL_CONFIGS = {
    1.5: {'window': 5, 'confidence': 0.70, 'min_confidence_bet': 0.55},  # External: AUC 0.627
    2.5: {'window': 5, 'confidence': 0.60, 'min_confidence_bet': 0.55},  # External: AUC 0.580 (was 0.509)
    3.5: {'window': 5, 'confidence': 0.65, 'min_confidence_bet': 0.55},  # External: AUC 0.586
    4.5: {'window': 5, 'confidence': 0.70, 'min_confidence_bet': 0.55},  # External: AUC 0.699
}

# Monitoring config for threshold 2.5 (most improved)
MONITORING_CONFIG = {
    'threshold_2_5': {
        'enabled': True,
        'log_file': os.path.join(os.path.dirname(__file__), '../logs/predictions_25.log'),
        'min_prob_spread': 0.1,  # Flag if |P(Over) - 0.5| < 0.1
    }
}

# Load models at startup
models = {}
goals_models = {}

# URL for downloading models zip (set via environment variable or use default)
MODELS_ZIP_URL = os.environ.get(
    'MODELS_ZIP_URL',
    'https://huggingface.co/qnguyen3/epl-corners-2526-v1/resolve/main/models.zip'
)


def download_and_extract_models():
    """Download and extract models zip from external storage if not present locally."""
    import zipfile
    from io import BytesIO

    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Check if models already exist
    if os.path.exists(model_dir) and any(f.endswith('.pkl') for f in os.listdir(model_dir)):
        print("Models already present, skipping download")
        return

    if not MODELS_ZIP_URL:
        print("MODELS_ZIP_URL not set, skipping model download")
        return

    print(f"Downloading models from {MODELS_ZIP_URL}...")
    try:
        response = requests.get(MODELS_ZIP_URL, timeout=300)
        response.raise_for_status()
        print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")

        # Extract zip
        os.makedirs(model_dir, exist_ok=True)
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(model_dir)
        print(f"Extracted models to {model_dir}")
    except Exception as e:
        print(f"Failed to download/extract models: {e}")


def load_models():
    """Load all classifier models from the models directory."""
    global models, goals_models
    # Reset to avoid accumulation across debug reloads
    models = {}
    goals_models = {}
    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Try to download models if not present
    download_and_extract_models()

    if not os.path.exists(model_dir):
        print(f"Warning: Models directory not found at {model_dir}")
        return

    # Load corner models (latest per threshold)
    corner_files = glob.glob(os.path.join(model_dir, 'ou*.pkl'))
    corner_latest: dict[float, str] = {}
    for f in corner_files:
        try:
            # Prefer filename pattern: ou{threshold}_...
            base = os.path.basename(f)
            # Fallback to mtime if parsing fails
            mtime = os.path.getmtime(f)
            th = None
            try:
                # Extract threshold from pickle for accuracy
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    th = float(data.get('threshold'))
            except Exception:
                pass
            if th is None:
                # Try parsing from filename like ou10.5_...
                import re
                m = re.search(r"ou([0-9]+\.[0-9]+|[0-9]+)", base)
                if m:
                    th = float(m.group(1))
            if th is None:
                continue
            prev = corner_latest.get(th)
            if not prev or os.path.getmtime(f) > os.path.getmtime(prev):
                corner_latest[th] = f
        except Exception as e:
            print(f"Error scanning corner model {f}: {e}")
    for th, f in corner_latest.items():
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                models[th] = data
                print(f"Loaded corner model for O/U {th}")
        except Exception as e:
            print(f"Error loading corner model {f}: {e}")

    # Load goals classifier models - PREFER EXTERNAL VERSION
    # Priority: external (with external data) > v3 (optimized) > v2 (baseline)
    goals_files = glob.glob(os.path.join(model_dir, 'goals_classifier_*.pkl'))
    goals_latest: dict[float, tuple[str, int, str]] = {}
    
    print(f"[*] Found {len(goals_files)} goals_classifier_*.pkl files")
    
    # Group by threshold, preferring external version
    for f in goals_files:
        try:
            base = os.path.basename(f).replace('.pkl', '')
            # Extract threshold and version from pattern: goals_classifier_{version}_{threshold}
            import re
            # Pattern: goals_classifier_external_1.5 or goals_classifier_v3_2.5
            match = re.search(r'goals_classifier_([a-zA-Z0-9]+)_([0-9.]+)$', base)
            if not match:
                print(f"    [!] Regex failed for: {base}")
                continue
            version = match.group(1)  # 'external', 'v3', 'v2', etc.
            threshold = float(match.group(2))
            
            # Priority score (higher = preferred)
            priority_map = {'external': 3, 'v3': 2, 'v2': 1}
            priority = priority_map.get(version, 0)
            
            if threshold not in goals_latest:
                goals_latest[threshold] = (f, priority, version)
            else:
                prev_f, prev_priority, prev_version = goals_latest[threshold]
                if priority > int(prev_priority) or (priority == int(prev_priority) and os.path.getmtime(f) > os.path.getmtime(prev_f)):
                    goals_latest[threshold] = (f, priority, version)
        except Exception as e:
            print(f"    [!] Error scanning {f}: {e}")
    
    print(f"[*] Selected {len(goals_latest)} external models for loading")
    
    for th, (f, priority, version) in goals_latest.items():
        try:
            # Use encoding='latin1' to handle models with Unicode comments
            with open(f, 'rb') as file:
                import sys
                # Try with UTF-8 first, fallback to latin1
                try:
                    file_content = file.read()
                    import io
                    data = pickle.loads(file_content)
                except (UnicodeDecodeError, pickle.UnpicklingError):
                    file.seek(0)
                    data = pickle.load(file, encoding='latin1')
                
                goals_models[th] = data
                print(f"✓ Loaded goals classifier for O/U {th} ({version}, AUC +14-24%)")
        except Exception as e:
            print(f"Error loading goals classifier {f}: {e}")
    
    # Also load latest regression model (fallback)
    reg_files = glob.glob(os.path.join(model_dir, 'model_v*_goals*.pkl'))
    latest_reg = None
    for f in reg_files:
        try:
            if not latest_reg or os.path.getmtime(f) > os.path.getmtime(latest_reg):
                latest_reg = f
        except Exception:
            pass
    if latest_reg:
        try:
            with open(latest_reg, 'rb') as file:
                data = pickle.load(file)
                version = data.get('version', 'unknown')
                goals_models['regression'] = data
                print(f"Loaded goals regression model ({version})")
        except Exception as e:
            print(f"Error loading goals regression model {latest_reg}: {e}")

# Load models on startup - wrap in try/except to prevent crash
try:
    logger.info("Loading models...")
    load_models()
    logger.info(f"Models loaded: {len(models)} corner models, {len(goals_models)} goals models")
except Exception as e:
    logger.error(f"Failed to load models on startup: {e}", exc_info=True)
    logger.warning("Running in degraded mode without models")


def fetch_data():
    """Fetch fresh data from football-data.co.uk (current season only)."""
    # Check cache first
    cache_key = f"data_{CURRENT_LEAGUE}_2526"
    cached_df = data_cache.get(cache_key)
    if cached_df is not None:
        return cached_df
    
    try:
        # Always use current league's URL, not the global DATA_URL
        current_url = get_data_url(CURRENT_LEAGUE, '2526')
        df = pd.read_csv(current_url, encoding='utf-8', on_bad_lines='skip')
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HC', 'AC'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['TotalCorners'] = df['HC'] + df['AC']
        
        # Cache the result
        data_cache.set(cache_key, df)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def fetch_historical_data():
    """Fetch multiple seasons for H2H analysis."""
    # Check cache first
    cache_key = f"historical_{CURRENT_LEAGUE}"
    cached_df = data_cache.get(cache_key)
    if cached_df is not None:
        return cached_df
    
    try:
        dfs = []
        for season in HISTORICAL_SEASONS:
            url = get_data_url(CURRENT_LEAGUE, season)
            try:
                df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
                df['Season'] = season
                dfs.append(df)
            except:
                pass
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HC', 'AC'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['TotalCorners'] = df['HC'] + df['AC']
        
        # Cache the result
        data_cache.set(cache_key, df)
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


def run_predictions(df, home_team, away_team):
    """Run predictions for all thresholds."""
    predictions = {}

    for threshold, config in MODEL_CONFIGS.items():
        window = config['window']
        confidence = config['confidence']

        # Get features for this match
        features = compute_features_for_match(df, home_team, away_team, window=window)

        # Check if model is loaded
        if threshold not in models:
            predictions[str(threshold)] = {
                'prob_over': 0.5,
                'prob_under': 0.5,
                'recommendation': 'LOW CONFIDENCE',
                'confidence_threshold': confidence,
                'error': 'Model not loaded'
            }
            continue

        model_data = models[threshold]
        model = model_data['model']
        feature_list = model_data['features']

        # Prepare feature vector (reindex ensures correct feature order and no KeyError)
        X = pd.DataFrame([features]).reindex(columns=feature_list, fill_value=0)

        # Make prediction
        try:
            prob_over = model.predict_proba(X)[0][1]
            prob_under = 1 - prob_over

            # Determine recommendation
            if prob_over > confidence:
                recommendation = 'OVER'
            elif prob_under > confidence:
                recommendation = 'UNDER'
            else:
                recommendation = 'LOW CONFIDENCE'

            predictions[str(threshold)] = {
                'prob_over': round(float(prob_over), 4),
                'prob_under': round(float(prob_under), 4),
                'recommendation': recommendation,
                'confidence_threshold': confidence,
            }
        except Exception as e:
            predictions[str(threshold)] = {
                'prob_over': 0.5,
                'prob_under': 0.5,
                'recommendation': 'LOW CONFIDENCE',
                'confidence_threshold': confidence,
                'error': str(e)
            }

    return predictions


def run_predictions_goals(df, home_team, away_team):
    """Run predictions for goals using classifiers (preferred) or regression model (fallback)."""
    predictions = {}

    # Check if classifiers are loaded
    has_classifiers = any(k in goals_models for k in [1.5, 2.5, 3.5, 4.5])
    
    if has_classifiers:
        # Use classifier approach
        for threshold, config in GOALS_MODEL_CONFIGS.items():
            if threshold not in goals_models:
                predictions[str(threshold)] = {
                    'prob_over': 0.5,
                    'prob_under': 0.5,
                    'recommendation': 'LOW CONFIDENCE',
                    'confidence_threshold': config['confidence'],
                    'error': f'Classifier for {threshold} not loaded'
                }
                continue
            
            model_data = goals_models[threshold]
            model = model_data['model']
            feature_list = model_data['features']
            optimal_threshold = model_data.get('threshold', 0.5)  # Changed from 'optimal_threshold'
            calibrator = model_data.get('calibrator')
            scaler = model_data.get('scaler')
            
            # Get features
            features = compute_features_for_goals_match(df, home_team, away_team, window=7)
            
            try:
                X = pd.DataFrame([features]).reindex(columns=feature_list, fill_value=0)
                
                # Apply scaler if available (external models use standardized features)
                if scaler is not None:
                    X = pd.DataFrame(scaler.transform(X), columns=feature_list)
                
                prob_over = model.predict_proba(X)[0][1]
                # Apply calibration if available
                if calibrator is not None:
                    try:
                        prob_over_cal = float(calibrator.predict([prob_over])[0])
                    except Exception:
                        prob_over_cal = float(prob_over)
                else:
                    prob_over_cal = float(prob_over)
                prob_under = 1.0 - prob_over_cal
                
                # Signal using optimal threshold
                signal = 'OVER' if prob_over_cal > optimal_threshold else ('UNDER' if (1 - prob_over_cal) > (1 - optimal_threshold) else 'NEUTRAL')

                # Recommendation based on confidence thresholds
                confidence = config['confidence']
                min_bet_confidence = config.get('min_confidence_bet', 0.55)
                
                if prob_over_cal >= confidence:
                    recommendation = 'OVER'
                    confidence_flag = 'HIGH' if prob_over_cal >= 0.65 else 'MEDIUM'
                elif prob_under >= confidence:
                    recommendation = 'UNDER'
                    confidence_flag = 'HIGH' if prob_under >= 0.65 else 'MEDIUM'
                else:
                    recommendation = 'LOW CONFIDENCE'
                    confidence_flag = 'LOW'
                
                # Additional flag: is confidence sufficient for betting?
                is_bettable = max(prob_over_cal, prob_under) >= min_bet_confidence
                
                predictions[str(threshold)] = {
                    'prob_over': round(float(prob_over_cal), 4),
                    'prob_under': round(float(prob_under), 4),
                    'recommendation': recommendation,
                    'signal': signal,
                    'confidence_threshold': confidence,
                    'confidence_level': confidence_flag,
                    'is_bettable': is_bettable,
                    'model_type': 'classifier',
                    'calibrated': calibrator is not None
                }
                
                # MONITORING: Log threshold 2.5 predictions
                if threshold == 2.5 and MONITORING_CONFIG['threshold_2_5']['enabled']:
                    try:
                        log_dir = os.path.dirname(MONITORING_CONFIG['threshold_2_5']['log_file'])
                        os.makedirs(log_dir, exist_ok=True)
                        prob_spread = abs(prob_over_cal - 0.5)
                        min_spread = MONITORING_CONFIG['threshold_2_5']['min_prob_spread']
                        with open(MONITORING_CONFIG['threshold_2_5']['log_file'], 'a') as log_f:
                            import datetime
                            now = datetime.datetime.now().isoformat()
                            log_f.write(f"{now}|{home_team}|{away_team}|{prob_over_cal:.4f}|{recommendation}|{confidence_flag}|spread={prob_spread:.3f}\n")
                    except Exception as e:
                        print(f"[!] Failed to log 2.5 prediction: {e}")
            except Exception as e:
                predictions[str(threshold)] = {
                    'prob_over': 0.5,
                    'prob_under': 0.5,
                    'recommendation': 'LOW CONFIDENCE',
                    'confidence_threshold': config['confidence'],
                    'confidence_level': 'LOW',
                    'is_bettable': False,
                    'error': str(e)
                }
    
    elif 'regression' in goals_models:
        # Fallback to regression model
        model_data = goals_models['regression']
        model = model_data['model']
        feature_list = model_data['feature_columns']
        
        features = compute_features_for_goals_match(df, home_team, away_team, window=5)
        
        try:
            X = pd.DataFrame([features])[feature_list]
            predicted_goals = model.predict(X)[0]
            
            for threshold, config in GOALS_MODEL_CONFIGS.items():
                confidence = config['confidence']
                prob_over = 1.0 / (1.0 + np.exp(-(predicted_goals - threshold) * 2))
                prob_under = 1.0 - prob_over
                
                if prob_over > confidence:
                    recommendation = 'OVER'
                elif prob_under > confidence:
                    recommendation = 'UNDER'
                else:
                    recommendation = 'LOW CONFIDENCE'
                
                predictions[str(threshold)] = {
                    'prob_over': round(float(prob_over), 4),
                    'prob_under': round(float(prob_under), 4),
                    'recommendation': recommendation,
                    'confidence_threshold': confidence,
                    'predicted_value': round(float(predicted_goals), 2),
                    'model_type': 'regression'
                }
        except Exception as e:
            for threshold in GOALS_MODEL_CONFIGS.keys():
                predictions[str(threshold)] = {
                    'prob_over': 0.5,
                    'prob_under': 0.5,
                    'recommendation': 'LOW CONFIDENCE',
                    'confidence_threshold': GOALS_MODEL_CONFIGS[threshold]['confidence'],
                    'error': str(e)
                }
    else:
        # No model loaded
        for threshold in GOALS_MODEL_CONFIGS.keys():
            predictions[str(threshold)] = {
                'prob_over': 0.5,
                'prob_under': 0.5,
                'recommendation': 'LOW CONFIDENCE',
                'confidence_threshold': GOALS_MODEL_CONFIGS[threshold]['confidence'],
                'error': 'No goals model loaded'
            }
    
    return predictions


def get_llm_assessment(home_team, away_team, predictions, statistics):
    """Get LLM assessment from OpenRouter API with configurable model + fallback."""
    if not OPENROUTER_API_KEY:
        return "LLM assessment not available (API key not configured)"

    # Build prompt
    prompt = f"""You are a sports betting analyst. Analyze this Premier League match and provide betting recommendations.

## Match: {home_team} vs {away_team}

## Model Predictions (Corner O/U)

| Threshold | P(Over) | P(Under) | Recommendation |
|-----------|---------|----------|----------------|
"""
    for threshold in ['8.5', '9.5', '10.5', '11.5', '12.5']:
        p = predictions.get(threshold, {})
        prompt += f"| O/U {threshold} | {p.get('prob_over', 0):.1%} | {p.get('prob_under', 0):.1%} | {p.get('recommendation', 'N/A')} |\n"

    # Add team statistics
    home_stats = statistics.get('home_team', {})
    away_stats = statistics.get('away_team', {})

    prompt += f"""

## {home_team} Recent Form (Home Games)
- Avg Corners For: {home_stats.get('avg_corners_for', 0):.1f}
- Avg Corners Against: {home_stats.get('avg_corners_against', 0):.1f}
- Avg Total Corners: {home_stats.get('avg_total_corners', 0):.1f}
- Over 9.5 Rate: {home_stats.get('over_rates', {}).get('9.5', 0):.0%}
- Over 10.5 Rate: {home_stats.get('over_rates', {}).get('10.5', 0):.0%}

## {away_team} Recent Form (Away Games)
- Avg Corners For: {away_stats.get('avg_corners_for', 0):.1f}
- Avg Corners Against: {away_stats.get('avg_corners_against', 0):.1f}
- Avg Total Corners: {away_stats.get('avg_total_corners', 0):.1f}
- Over 9.5 Rate: {away_stats.get('over_rates', {}).get('9.5', 0):.0%}
- Over 10.5 Rate: {away_stats.get('over_rates', {}).get('10.5', 0):.0%}

## Head-to-Head
"""
    h2h = statistics.get('head_to_head', {})
    if h2h.get('matches'):
        prompt += f"- Avg Total Corners in H2H: {h2h.get('avg_total_corners', 0):.1f}\n"
        prompt += f"- {home_team} Wins: {h2h.get('home_team_wins', 0)}, {away_team} Wins: {h2h.get('away_team_wins', 0)}, Draws: {h2h.get('draws', 0)}\n"
    else:
        prompt += "- No recent head-to-head data available\n"

    prompt += """

## Important Notes
- Standard odds are around 1.91 (-110), requiring 52.4% win rate to break even
- Higher confidence thresholds indicate stronger model signals

---

## INSTRUCTIONS

You MUST follow this EXACT output format. Do not deviate from this structure.

---

## OUTPUT FORMAT (Follow Exactly)

### 🎯 PRIMARY RECOMMENDATION

**Bet:** [OVER/UNDER] [threshold] corners
**Confidence:** [HIGH/MEDIUM/LOW]

### 📊 MATCH ANALYSIS

[2-3 sentences analyzing key factors: team form, playing styles, tactical tendencies that affect corner count]

### 📈 MODEL INSIGHTS

| Threshold | Signal | Strength |
|-----------|--------|----------|
| O/U 8.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 9.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 10.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 11.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 12.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |

### ⚠️ RISK FACTORS

- [Risk factor 1]
- [Risk factor 2]
- [Risk factor 3 if applicable]

### 🔄 ALTERNATIVE BETS

1. **[Bet 1]:** [Brief reasoning]
2. **[Bet 2]:** [Brief reasoning]

### 💡 VERDICT

[1-2 sentence final summary with clear action recommendation]

---

IMPORTANT: Follow the exact format above. Use the exact headers with emojis. Keep each section concise. Do not add extra sections."""

    def _call_openrouter(model_name, prompt_text):
        response = requests.post(
            OPENROUTER_URL,
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://epl-scope.vercel.app',
                'X-Title': 'EPL SCOPE'
            },
            json={
                'model': model_name,
                'messages': [
                    {'role': 'user', 'content': prompt_text}
                ]
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"
    
    tried_models = []
    errors = []
    # Use a default model if none configured
    model_queue = ['meta-llama/llama-2-70b-chat:free']

    for model_name in model_queue:
        tried_models.append(model_name)
        try:
            data = _call_openrouter(model_name, prompt)
            return data['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else ''
            if status == 429:
                errors.append(f"{model_name}: 429 rate limit/quota ({body[:120]}...)")
                continue
            errors.append(f"{model_name}: HTTP {status} ({body[:120]}...)")
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")

    return (
        "LLM assessment failed: all models exhausted. "
        f"Tried: {', '.join(tried_models)}. Errors: " + " | ".join(errors)
    )


def get_llm_assessment_goals(home_team, away_team, predictions, statistics):
    """Get LLM assessment focused on Goals O/U."""
    if not OPENROUTER_API_KEY:
        return "LLM assessment (goals) not available (API key not configured)"

    prompt = f"""You are a sports betting analyst. Analyze this Premier League match and provide betting recommendations.

## Match: {home_team} vs {away_team}

## Model Predictions (Goals O/U)

| Threshold | P(Over) | P(Under) | Recommendation |
|-----------|---------|----------|----------------|
"""
    for threshold in ['1.5', '2.5', '3.5', '4.5']:
        p = predictions.get(threshold, {})
        prompt += f"| O/U {threshold} | {p.get('prob_over', 0):.1%} | {p.get('prob_under', 0):.1%} | {p.get('recommendation', 'N/A')} |\n"

    home_stats = statistics.get('home_team', {})
    away_stats = statistics.get('away_team', {})

    prompt += f"""

## {home_team} Recent Form (Home Games)
- Avg Shots For: {home_stats.get('avg_shots_for', 0):.1f}
- Avg Shots Against: {home_stats.get('avg_shots_against', 0):.1f}
- Avg Goals For: {home_stats.get('avg_goals_for', 0):.1f}
- Avg Goals Against: {home_stats.get('avg_goals_against', 0):.1f}

## {away_team} Recent Form (Away Games)
- Avg Shots For: {away_stats.get('avg_shots_for', 0):.1f}
- Avg Shots Against: {away_stats.get('avg_shots_against', 0):.1f}
- Avg Goals For: {away_stats.get('avg_goals_for', 0):.1f}
- Avg Goals Against: {away_stats.get('avg_goals_against', 0):.1f}

## Notes
- Typical O/U odds ~1.91 (-110), break-even 52.4%
- Higher confidence thresholds indicate stronger model signals

---

## INSTRUCTIONS

You MUST follow this EXACT output format. Do not deviate.

---

## OUTPUT FORMAT (Follow Exactly)

### 🎯 PRIMARY RECOMMENDATION

**Bet:** [OVER/UNDER] [threshold] goals
**Confidence:** [HIGH/MEDIUM/LOW]

### 📊 MATCH ANALYSIS

[2-3 sentences: playing styles, finishing quality, defensive solidity affecting total goals]

### 📈 MODEL INSIGHTS

| Threshold | Signal | Strength |
|-----------|--------|----------|
| O/U 1.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 2.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 3.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |
| O/U 4.5 | [OVER/UNDER/NEUTRAL] | [Strong/Moderate/Weak] |

### ⚠️ RISK FACTORS

- [Risk 1]
- [Risk 2]
- [Risk 3]

### 🔄 ALTERNATIVE BETS

1. **[Bet 1]:** [Brief reasoning]
2. **[Bet 2]:** [Brief reasoning]

### 💡 VERDICT

[1-2 sentence final summary]

---

IMPORTANT: Follow the exact format above.
"""

    def _call_openrouter(model_name, prompt_text):
        response = requests.post(
            OPENROUTER_URL,
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://epl-scope.vercel.app',
                'X-Title': 'EPL SCOPE'
            },
            json={
                'model': model_name,
                'messages': [
                    {'role': 'user', 'content': prompt_text}
                ]
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"
    
    tried_models = []
    errors = []
    # Use a default model if none configured
    model_queue = ['meta-llama/llama-2-70b-chat:free']

    for model_name in model_queue:
        tried_models.append(model_name)
        try:
            data = _call_openrouter(model_name, prompt)
            return data['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else ''
            if status == 429:
                errors.append(f"{model_name}: 429 rate limit/quota ({body[:120]}...)")
                continue
            errors.append(f"{model_name}: HTTP {status} ({body[:120]}...)")
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")

    return (
        "LLM assessment (goals) failed: all models exhausted. "
        f"Tried: {', '.join(tried_models)}. Errors: " + " | ".join(errors)
    )


def prepare_charts_data(df, home_team, away_team, predictions):
    """Prepare data for frontend charts."""
    # Home team recent corner history
    home_matches = df[df['HomeTeam'] == home_team].tail(10)
    home_corner_history = []
    for _, row in home_matches.iterrows():
        home_corner_history.append({
            'date': row['Date'].strftime('%m/%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'opponent': row['AwayTeam'],
            'corners_for': int(row['HC']),
            'corners_against': int(row['AC']),
            'total': int(row['HC'] + row['AC'])
        })

    # Away team recent corner history
    away_matches = df[df['AwayTeam'] == away_team].tail(10)
    away_corner_history = []
    for _, row in away_matches.iterrows():
        away_corner_history.append({
            'date': row['Date'].strftime('%m/%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'opponent': row['HomeTeam'],
            'corners_for': int(row['AC']),
            'corners_against': int(row['HC']),
            'total': int(row['HC'] + row['AC'])
        })

    # Probability by threshold for line chart
    prob_by_threshold = []
    for threshold in ['8.5', '9.5', '10.5', '11.5', '12.5']:
        p = predictions.get(threshold, {})
        prob_by_threshold.append({
            'threshold': threshold,
            'prob_over': p.get('prob_over', 0.5),
            'prob_under': p.get('prob_under', 0.5),
        })

    return {
        'home_corner_history': home_corner_history,
        'away_corner_history': away_corner_history,
        'probability_by_threshold': prob_by_threshold
    }


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/health')
def health_root():
    """Root health check endpoint for Railway."""
    return jsonify({'status': 'ok'}), 200

@app.route('/api/health')
@handle_errors
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'thresholds': sorted(list(models.keys())),
        'current_league': CURRENT_LEAGUE,
        'cache_size': len(data_cache.cache),
        'timestamp': time.time()
    })


@app.route('/api/cache/clear', methods=['POST'])
@handle_errors
def clear_cache():
    """Clear data cache (admin endpoint)."""
    data_cache.clear()
    logger.warning("Cache cleared by request")
    return jsonify({'message': 'Cache cleared successfully', 'status': 'success'})


@app.route('/api/leagues')
@handle_errors
def get_leagues():
    """Get list of available leagues."""
    response = jsonify({
        'leagues': [
            {
                'code': code,
                'name': info['name'],
                'country': info['country'],
                'flag': info['flag'],
                'iso': info['iso']
            }
            for code, info in LEAGUES.items()
        ],
        'current': CURRENT_LEAGUE,
        'count': len(LEAGUES)
    })
    response.cache_control.max_age = 3600  # Cache for 1 hour
    response.cache_control.public = True
    return response


@app.route('/api/league/current')
@handle_errors
def get_current_league():
    """Get currently selected league."""
    return jsonify({
        'code': CURRENT_LEAGUE,
        'name': LEAGUES[CURRENT_LEAGUE]['name'],
        'country': LEAGUES[CURRENT_LEAGUE]['country'],
        'flag': LEAGUES[CURRENT_LEAGUE]['flag'],
        'iso': LEAGUES[CURRENT_LEAGUE]['iso']
    })


@app.route('/api/league/select', methods=['POST'])
def select_league():
    """Select a league to use for predictions."""
    global CURRENT_LEAGUE, DATA_URL
    
    try:
        data = request.get_json()
        league_code = data.get('league_code', '').upper()
        
        if not league_code:
            return jsonify({'error': 'league_code is required'}), 400
        
        if league_code not in LEAGUES:
            return jsonify({
                'error': f'Invalid league code. Available: {list(LEAGUES.keys())}'
            }), 400
        
        # Update global state
        CURRENT_LEAGUE = league_code
        DATA_URL = get_data_url(league_code, '2526')
        
        # Reload models for the new league
        logger.info(f"Reloading models for league {league_code}")
        load_models()
        logger.info(f"Models reloaded: {len(models)} corner models, {len(goals_models)} goals models")
        
        return jsonify({
            'success': True,
            'league': {
                'code': CURRENT_LEAGUE,
                'name': LEAGUES[CURRENT_LEAGUE]['name'],
                'country': LEAGUES[CURRENT_LEAGUE]['country'],
                'flag': LEAGUES[CURRENT_LEAGUE]['flag'],
                'iso': LEAGUES[CURRENT_LEAGUE]['iso']
            },
            'message': f'Switched to {LEAGUES[CURRENT_LEAGUE]["name"]}',
            'models_loaded': {
                'corners': len(models),
                'goals': len(goals_models)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/teams')
@handle_errors
def get_teams():
    """Get list of current season teams."""
    df = fetch_data()
    if df is None:
        raise ValueError('Failed to fetch data from source')

    teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    response = jsonify({
        'teams': teams,
        'league': CURRENT_LEAGUE,
        'count': len(teams)
    })
    response.cache_control.max_age = 600  # Cache for 10 minutes
    response.cache_control.public = True
    response.headers['Vary'] = 'Cookie'
    return response


@app.route('/api/predict', methods=['POST'])
@handle_errors
def predict():
    """Run predictions for a match."""
    data = request.json or {}
    home_team = validate_team(data.get('home_team', ''))
    away_team = validate_team(data.get('away_team', ''))

    if home_team == away_team:
        raise ValueError('Home and away teams must be different')

    # Fetch fresh data
    df = fetch_data()
    if df is None:
        raise ValueError('Failed to fetch data from source')

    # Validate teams
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    if home_team not in all_teams:
        raise ValueError(f'Unknown team: {home_team}. Available teams: {", ".join(sorted(all_teams))}')
    if away_team not in all_teams:
        raise ValueError(f'Unknown team: {away_team}. Available teams: {", ".join(sorted(all_teams))}')

    # Run predictions
    logger.info(f"Running predictions for: {home_team} vs {away_team}")
    predictions = run_predictions(df, home_team, away_team)

    # Fetch historical data for H2H (last 3 seasons)
    df_historical = fetch_historical_data()

    # Get statistics (use current season for team stats, historical for H2H)
    statistics = {
        'home_team': get_team_statistics(df, home_team),
        'away_team': get_team_statistics(df, away_team),
        'head_to_head': get_head_to_head(df_historical if df_historical is not None else df, home_team, away_team)
    }

    # Get LLM assessment
    llm_assessment = get_llm_assessment(home_team, away_team, predictions, statistics)

    # Prepare charts data
    charts_data = prepare_charts_data(df, home_team, away_team, predictions)

    return jsonify({
        'match': {
            'home_team': home_team,
            'away_team': away_team
        },
        'predictions': predictions,
        'statistics': statistics,
        'llm_assessment': llm_assessment,
        'charts_data': charts_data
    })


@app.route('/api/predict/goals', methods=['POST'])
@handle_errors
def predict_goals():
    """Run predictions for Total Goals in a match."""
    data = request.json or {}
    home_team = validate_team(data.get('home_team', ''))
    away_team = validate_team(data.get('away_team', ''))

    if home_team == away_team:
        raise ValueError('Home and away teams must be different')

    # Fetch fresh data
    df = fetch_data()
    if df is None:
        raise ValueError('Failed to fetch data from source')

    # Validate teams
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    if home_team not in all_teams:
        raise ValueError(f'Unknown team: {home_team}')
    if away_team not in all_teams:
        raise ValueError(f'Unknown team: {away_team}')

    # Run goals predictions
    logger.info(f"Running goals predictions for: {home_team} vs {away_team}")
    predictions = run_predictions_goals(df, home_team, away_team)

    # Get statistics
    statistics = {
        'home_team': get_team_statistics(df, home_team),
        'away_team': get_team_statistics(df, away_team),
    }

    return jsonify({
        'match': {
            'home_team': home_team,
            'away_team': away_team,
            'prediction_type': 'goals'
        },
        'predictions': predictions,
        'statistics': statistics
    })


@app.route('/api/predict/all', methods=['POST'])
def predict_all():
    """Run predictions for BOTH corners AND goals in a single request."""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        include_llm = bool(data.get('include_llm', True))

        if not home_team or not away_team:
            return jsonify({'error': 'home_team and away_team are required'}), 400

        if home_team == away_team:
            return jsonify({'error': 'Home and away teams must be different'}), 400

        # Fetch fresh data
        df = fetch_data()
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 500

        # Validate teams
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        if home_team not in all_teams:
            return jsonify({'error': f'Unknown team: {home_team}'}), 400
        if away_team not in all_teams:
            return jsonify({'error': f'Unknown team: {away_team}'}), 400

        # Run BOTH predictions
        print(f"[*] Analyzing match (ALL): {home_team} vs {away_team}")
        corners_predictions = run_predictions(df, home_team, away_team)
        goals_predictions = run_predictions_goals(df, home_team, away_team)

        # Fetch historical data for H2H
        df_historical = fetch_historical_data()

        # Get statistics
        statistics = {
            'home_team': get_team_statistics(df, home_team),
            'away_team': get_team_statistics(df, away_team),
            'head_to_head': get_head_to_head(df_historical if df_historical is not None else df, home_team, away_team)
        }

        # Get LLM assessments (optional)
        if include_llm:
            llm_assessment = get_llm_assessment(home_team, away_team, corners_predictions, statistics)
            # Goals LLM removed: not needed, corners analysis covers both
            llm_assessment_goals = None
        else:
            llm_assessment = "LLM disabled for this request"
            llm_assessment_goals = None

        # Prepare charts data
        charts_data = prepare_charts_data(df, home_team, away_team, corners_predictions)

        return jsonify({
            'match': {
                'home_team': home_team,
                'away_team': away_team,
                'prediction_types': ['corners', 'goals']
            },
            'corners': {
                'predictions': corners_predictions,
                'thresholds': ['8.5', '9.5', '10.5', '11.5', '12.5']
            },
            'goals': {
                'predictions': goals_predictions,
                'thresholds': ['1.5', '2.5', '3.5', '4.5']
            },
            'statistics': statistics,
            'llm_assessment': llm_assessment,
            'llm_assessment_goals': llm_assessment_goals,
            'charts_data': charts_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
