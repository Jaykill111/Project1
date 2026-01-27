"""
OPTIMIZED: Goals Classifiers Training Script V2
Fixes for Threshold 2.5:
1. Remove redundant xG features (perfect correlation with SOT)
2. Add recent form features (3-game and 7-game trends)
3. Apply SMOTE for class imbalance
4. Implement outlier capping at 95th percentile
5. Better temporal cross-validation
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[!] SMOTE not available - install: pip install imbalanced-learn")

# =============================================================================
# CONFIGURATION
# =============================================================================

ROLLING_WINDOW = 7
TEST_SEASON = '2025-26'
THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

# LightGBM parameters - ADJUSTED for better generalization
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 25,  # Reduced from 31 to prevent overfitting
    'max_depth': 5,    # Reduced from 6
    'learning_rate': 0.03,  # Reduced from 0.05
    'n_estimators': 400,  # Increased iterations
    'min_child_samples': 25,  # Increased from 20
    'subsample': 0.75,  # Reduced from 0.8
    'colsample_bytree': 0.75,  # Reduced from 0.8
    'reg_alpha': 0.2,  # Increased regularization
    'reg_lambda': 0.8,  # Increased regularization
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

EARLY_STOPPING_ROUNDS = 60

# =============================================================================
# DATA LOADING
# =============================================================================
print("="*70)
print("SCOPE GOALS CLASSIFIERS TRAINING V2 (OPTIMIZED)")
print("="*70)

SEASONS = {
    '2020-21': '2021',
    '2021-22': '2122',
    '2022-23': '2223',
    '2023-24': '2324',
    '2024-25': '2425',
    '2025-26': '2526'
}

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'
COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']

print("\nLoading data...")
dfs = []
for season_name, season_code in SEASONS.items():
    url = BASE_URL.format(code=season_code)
    try:
        df = pd.read_csv(url, encoding='utf-8')
        available_cols = [c for c in COLS if c in df.columns]
        df = df[available_cols].copy()
        df['Season'] = season_name
        dfs.append(df)
        print(f"  {season_name}: {len(df)} matches")
    except Exception as e:
        print(f"  {season_name}: Failed - {e}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
df['TotalGoals'] = df['FTHG'] + df['FTAG']

print(f"\nTotal: {len(df)} matches")

# =============================================================================
# FEATURE ENGINEERING (OPTIMIZED)
# =============================================================================
print("\nComputing features...")

def compute_rolling_features_goals_v2(df, n=7):
    """
    Compute venue-aware rolling features for goals prediction.
    V2: Includes recent form features and volatility indicators.
    """
    rolling_cols = [
        'home_goals_for', 'home_goals_against', 'home_goals_std',
        'away_goals_for', 'away_goals_against', 'away_goals_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_corners_for', 'away_corners_for',
        # NEW: Recent form (3-game window)
        'home_goals_for_3', 'away_goals_for_3',
        'home_goals_for_trend', 'away_goals_for_trend',
    ]

    for col in rolling_cols:
        df[col] = np.nan

    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in all_teams:
        home_mask = df['HomeTeam'] == team
        away_mask = df['AwayTeam'] == team
        home_indices = df[home_mask].index.tolist()
        away_indices = df[away_mask].index.tolist()

        # HOME games
        for i, idx in enumerate(home_indices):
            if i >= n:
                prev = home_indices[i-n:i]
                prev_data = df.loc[prev]
                df.loc[idx, 'home_goals_for'] = prev_data['FTHG'].mean()
                df.loc[idx, 'home_goals_against'] = prev_data['FTAG'].mean()
                df.loc[idx, 'home_goals_std'] = prev_data['FTHG'].std()
                df.loc[idx, 'home_shots_for'] = prev_data['HS'].mean()
                df.loc[idx, 'home_shots_against'] = prev_data['AS'].mean()
                df.loc[idx, 'home_sot_for'] = prev_data['HST'].mean()
                df.loc[idx, 'home_sot_against'] = prev_data['AST'].mean()
                df.loc[idx, 'home_corners_for'] = prev_data['HC'].mean()
                
                # NEW: 3-game form
                if i >= 3:
                    prev_3 = home_indices[i-3:i]
                    prev_data_3 = df.loc[prev_3]
                    df.loc[idx, 'home_goals_for_3'] = prev_data_3['FTHG'].mean()
                    # Trend: recent vs older
                    first_half = df.loc[prev_3[0], 'FTHG'] if len(prev_3) > 0 else 0
                    df.loc[idx, 'home_goals_for_trend'] = prev_data_3['FTHG'].mean() - first_half

        # AWAY games
        for i, idx in enumerate(away_indices):
            if i >= n:
                prev = away_indices[i-n:i]
                prev_data = df.loc[prev]
                df.loc[idx, 'away_goals_for'] = prev_data['FTAG'].mean()
                df.loc[idx, 'away_goals_against'] = prev_data['FTHG'].mean()
                df.loc[idx, 'away_goals_std'] = prev_data['FTAG'].std()
                df.loc[idx, 'away_shots_for'] = prev_data['AS'].mean()
                df.loc[idx, 'away_shots_against'] = prev_data['HS'].mean()
                df.loc[idx, 'away_sot_for'] = prev_data['AST'].mean()
                df.loc[idx, 'away_sot_against'] = prev_data['HST'].mean()
                df.loc[idx, 'away_corners_for'] = prev_data['AC'].mean()
                
                # NEW: 3-game form
                if i >= 3:
                    prev_3 = away_indices[i-3:i]
                    prev_data_3 = df.loc[prev_3]
                    df.loc[idx, 'away_goals_for_3'] = prev_data_3['FTAG'].mean()
                    first_half = df.loc[prev_3[0], 'FTAG'] if len(prev_3) > 0 else 0
                    df.loc[idx, 'away_goals_for_trend'] = prev_data_3['FTAG'].mean() - first_half

    return df

df = compute_rolling_features_goals_v2(df, n=ROLLING_WINDOW)

# Derived features
print("Computing derived features...")
df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
df['home_goal_balance'] = df['home_goals_for'] - df['home_goals_against']
df['away_goal_balance'] = df['away_goals_for'] - df['away_goals_against']
df['goal_differential'] = df['home_goal_balance'] + df['away_goal_balance']

# Shot efficiency
df['home_shot_accuracy'] = df['home_sot_for'] / (df['home_shots_for'] + 0.001)
df['away_shot_accuracy'] = df['away_sot_for'] / (df['away_shots_for'] + 0.001)
df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']

# Conversion rate
df['home_conversion_rate'] = df['home_goals_for'] / (df['home_sot_for'] + 0.001)
df['away_conversion_rate'] = df['away_goals_for'] / (df['away_sot_for'] + 0.001)

# REMOVED: xG_proxy (perfect correlation with SOT)
# df['home_xg_proxy'] = df['home_sot_for'] * 0.35  # REDUNDANT
# df['away_xg_proxy'] = df['away_sot_for'] * 0.35  # REDUNDANT
# df['combined_xg_proxy'] = df['home_xg_proxy'] + df['away_xg_proxy']  # REDUNDANT

# Attack vs defense
df['attack_strength_diff'] = df['home_goals_for'] - df['away_goals_against']
df['defense_quality_diff'] = df['away_goals_for'] - df['home_goals_against']

# Volatility
df['home_goals_cv'] = df['home_goals_std'] / (df['home_goals_for'] + 0.001)
df['away_goals_cv'] = df['away_goals_std'] / (df['away_goals_for'] + 0.001)

# NEW: Recent momentum
df['home_momentum'] = df['home_goals_for_3'] - df['home_goals_for']
df['away_momentum'] = df['away_goals_for_3'] - df['away_goals_for']

# NEW: Combining both team trends
df['combined_momentum'] = df['home_momentum'] + df['away_momentum']

df = df.fillna(0)

# =============================================================================
# FEATURE SELECTION (OPTIMIZED - REMOVED REDUNDANT FEATURES)
# =============================================================================
FEATURE_COLUMNS = [
    # Core stats
    'home_goals_for', 'home_goals_against', 'home_goals_std',
    'away_goals_for', 'away_goals_against', 'away_goals_std',
    'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
    'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
    'home_corners_for', 'away_corners_for',
    # Derived
    'combined_goals_for', 'combined_goals_against',
    'home_goal_balance', 'away_goal_balance', 'goal_differential',
    'home_shot_accuracy', 'away_shot_accuracy',
    'combined_shots_for', 'combined_sot_for',
    'home_conversion_rate', 'away_conversion_rate',
    # xG removed (redundant)
    'attack_strength_diff', 'defense_quality_diff',
    'home_goals_cv', 'away_goals_cv',
    # NEW: Recent form features
    'home_goals_for_3', 'away_goals_for_3',
    'home_momentum', 'away_momentum', 'combined_momentum',
]

print(f"Features: {len(FEATURE_COLUMNS)} (reduced from 34)")

# =============================================================================
# TRAIN CLASSIFIERS FOR EACH THRESHOLD
# =============================================================================
df_model = df.dropna(subset=FEATURE_COLUMNS + ['TotalGoals']).copy()
print(f"Matches with complete features: {len(df_model)}")

train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

print(f"\nTraining: {len(train_df)} | Test: {len(test_df)}")

X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

# CAP OUTLIERS at 95th percentile
print("\nCapping outliers at 95th percentile...")
for feat in FEATURE_COLUMNS:
    p95 = X_train[feat].quantile(0.95)
    p5 = X_train[feat].quantile(0.05)
    X_train[feat] = X_train[feat].clip(p5, p95)
    X_test[feat] = X_test[feat].clip(p5, p95)
    print(f"  {feat:30s}: [{p5:.3f}, {p95:.3f}]")

results = []

for threshold in THRESHOLDS:
    print("\n" + "="*70)
    print(f"TRAINING CLASSIFIER: O/U {threshold}")
    print("="*70)
    
    # Create binary labels
    y_train = (train_df['TotalGoals'] > threshold).astype(int)
    y_test = (test_df['TotalGoals'] > threshold).astype(int)
    
    # Class distribution
    train_over_pct = y_train.mean() * 100
    test_over_pct = y_test.mean() * 100
    print(f"\nClass distribution (BEFORE SMOTE):")
    print(f"  Train - Over: {train_over_pct:.1f}% | Under: {100-train_over_pct:.1f}%")
    print(f"  Test  - Over: {test_over_pct:.1f}% | Under: {100-test_over_pct:.1f}%")
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train/val split (temporal)
    val_size = int(len(X_train) * 0.15)
    X_train_fit = X_train.iloc[:-val_size].copy()
    y_train_fit = y_train.iloc[:-val_size].copy()
    X_val = X_train.iloc[-val_size:].copy()
    y_val = y_train.iloc[-val_size:].copy()
    
    # SMOTE: Apply only to training set
    if HAS_SMOTE and threshold == 2.5:  # Apply to most problematic threshold
        print("\n✨ Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        try:
            X_train_fit_smote, y_train_fit_smote = smote.fit_resample(X_train_fit, y_train_fit)
            X_train_fit = pd.DataFrame(X_train_fit_smote, columns=FEATURE_COLUMNS)
            y_train_fit = pd.Series(y_train_fit_smote)
            train_over_pct_after = y_train_fit.mean() * 100
            print(f"  After SMOTE - Over: {train_over_pct_after:.1f}% | Under: {100-train_over_pct_after:.1f}%")
        except Exception as e:
            print(f"  SMOTE failed: {e}")
    
    # Adjust params
    params = LIGHTGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    y_val_proba = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.3, 0.7, 0.05):
        y_val_pred = (y_val_proba > t).astype(int)
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_threshold = t
    
    print(f"  Optimal prediction threshold: {best_threshold:.2f}")
    
    y_pred = (y_pred_proba > best_threshold).astype(int)
    
    # Calibration
    try:
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_val_proba, y_val)
        y_pred_proba_cal = calibrator.predict(y_pred_proba)
    except Exception:
        calibrator = None
        y_pred_proba_cal = y_pred_proba

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1-Score:  {f1:.1%}")
    print(f"  AUC-ROC:   {auc:.3f}")
    if calibrator is not None:
        try:
            auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
            print(f"  AUC-ROC (calibrated): {auc_cal:.3f}")
        except Exception:
            pass

    # Save model
    model_filename = f"api/models/goals_classifier_v2_{threshold}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved: {model_filename}")
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
print("\n✅ Training complete!")
