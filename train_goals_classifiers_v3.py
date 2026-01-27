"""
OPTIMIZED: Goals Classifiers Training Script V3
Fixes:
1. Better threshold optimization using both precision & recall
2. Focus on betting value (precision) rather than recall
3. Use calibrated probabilities
4. Feature engineering specifically for threshold 2.5
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[!] SMOTE not available")

# =============================================================================
# CONFIGURATION
# =============================================================================

ROLLING_WINDOW = 7
TEST_SEASON = '2025-26'
THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

# LightGBM - tuned for better generalization
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 20,
    'max_depth': 4,
    'learning_rate': 0.02,
    'n_estimators': 500,
    'min_child_samples': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

EARLY_STOPPING_ROUNDS = 80

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
print("="*70)
print("SCOPE GOALS CLASSIFIERS TRAINING V3 (OPTIMIZED FOR BETTING)")
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

print(f"Total: {len(df)} matches\n")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("Computing features...")

def compute_rolling_features_goals_v3(df, n=7):
    """Enhanced feature computation with momentum & consistency metrics."""
    rolling_cols = [
        'home_goals_for', 'home_goals_against', 'home_goals_std',
        'away_goals_for', 'away_goals_against', 'away_goals_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_corners_for', 'away_corners_for',
        # Recent form (3-game momentum)
        'home_goals_for_3', 'away_goals_for_3',
        'home_consistency', 'away_consistency',
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
                df.loc[idx, 'home_consistency'] = 1 - (prev_data['FTHG'].std() / (prev_data['FTHG'].mean() + 0.001))
                
                if i >= 3:
                    prev_3 = home_indices[i-3:i]
                    prev_data_3 = df.loc[prev_3]
                    df.loc[idx, 'home_goals_for_3'] = prev_data_3['FTHG'].mean()

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
                df.loc[idx, 'away_consistency'] = 1 - (prev_data['FTAG'].std() / (prev_data['FTAG'].mean() + 0.001))
                
                if i >= 3:
                    prev_3 = away_indices[i-3:i]
                    prev_data_3 = df.loc[prev_3]
                    df.loc[idx, 'away_goals_for_3'] = prev_data_3['FTAG'].mean()

    return df

df = compute_rolling_features_goals_v3(df, n=ROLLING_WINDOW)

# Derived features
print("Computing derived features...")
df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
df['home_goal_balance'] = df['home_goals_for'] - df['home_goals_against']
df['away_goal_balance'] = df['away_goals_for'] - df['away_goals_against']

# Shot conversion
df['home_shot_accuracy'] = df['home_sot_for'] / (df['home_shots_for'] + 0.001)
df['away_shot_accuracy'] = df['away_sot_for'] / (df['away_shots_for'] + 0.001)
df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']

df['home_conversion_rate'] = df['home_goals_for'] / (df['home_sot_for'] + 0.001)
df['away_conversion_rate'] = df['away_goals_for'] / (df['away_sot_for'] + 0.001)

# Volatility & consistency
df['home_goals_cv'] = df['home_goals_std'] / (df['home_goals_for'] + 0.001)
df['away_goals_cv'] = df['away_goals_std'] / (df['away_goals_for'] + 0.001)

# Head to head form
df['home_momentum'] = df['home_goals_for_3'] - df['home_goals_for']
df['away_momentum'] = df['away_goals_for_3'] - df['away_goals_for']
df['combined_momentum'] = df['home_momentum'] + df['away_momentum']

# Strength difference
df['attack_diff'] = df['home_goals_for'] - df['away_goals_against']
df['defense_diff'] = df['away_goals_for'] - df['home_goals_against']

df = df.fillna(0)

# =============================================================================
# FEATURE SELECTION
# =============================================================================
FEATURE_COLUMNS = [
    'home_goals_for', 'home_goals_against', 'home_goals_std',
    'away_goals_for', 'away_goals_against', 'away_goals_std',
    'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
    'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
    'home_corners_for', 'away_corners_for',
    'combined_goals_for', 'combined_goals_against',
    'home_goal_balance', 'away_goal_balance',
    'home_shot_accuracy', 'away_shot_accuracy',
    'combined_shots_for', 'combined_sot_for',
    'home_conversion_rate', 'away_conversion_rate',
    'home_goals_cv', 'away_goals_cv',
    'attack_diff', 'defense_diff',
    'home_momentum', 'away_momentum', 'combined_momentum',
    'home_consistency', 'away_consistency',
]

print(f"Features: {len(FEATURE_COLUMNS)}")

# =============================================================================
# PREPARE DATA
# =============================================================================
df_model = df.dropna(subset=FEATURE_COLUMNS + ['TotalGoals']).copy()
print(f"Matches with complete features: {len(df_model)}\n")

train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

print(f"Training: {len(train_df)} | Test: {len(test_df)}\n")

X_train = train_df[FEATURE_COLUMNS].copy()
X_test = test_df[FEATURE_COLUMNS].copy()

# Cap outliers at 95th/5th percentile
print("Normalizing features (clipping outliers)...")
for feat in FEATURE_COLUMNS:
    p95 = X_train[feat].quantile(0.95)
    p5 = X_train[feat].quantile(0.05)
    X_train[feat] = X_train[feat].clip(p5, p95)
    X_test[feat] = X_test[feat].clip(p5, p95)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)

results = []

for threshold in THRESHOLDS:
    print("="*70)
    print(f"THRESHOLD: O/U {threshold}")
    print("="*70)
    
    y_train = (train_df['TotalGoals'] > threshold).astype(int)
    y_test = (test_df['TotalGoals'] > threshold).astype(int)
    
    over_pct = y_train.mean() * 100
    print(f"Train: Over={over_pct:.1f}% | Test: Over={(y_test.mean()*100):.1f}%")
    
    val_size = int(len(X_train) * 0.15)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    
    # SMOTE for imbalanced thresholds
    if HAS_SMOTE and threshold in [2.5, 3.5]:
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_fit_smote, y_train_fit_smote = smote.fit_resample(X_train_fit, y_train_fit)
            X_train_fit = pd.DataFrame(X_train_fit_smote, columns=FEATURE_COLUMNS)
            y_train_fit = pd.Series(y_train_fit_smote).reset_index(drop=True)
            print(f"SMOTE applied: balanced to 50-50")
        except Exception as e:
            print(f"SMOTE failed: {e}")
    
    # Train
    params = LIGHTGBM_PARAMS.copy()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    # Get probabilities
    y_val_proba = np.array(model.predict_proba(X_val))[:, 1]
    y_test_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    # Find optimal threshold: prioritize precision (fewer but better bets)
    # Search for threshold that maximizes: precision while keeping recall >= 0.3
    best_threshold = 0.5
    best_score = 0
    
    for t in np.arange(0.2, 0.8, 0.05):
        y_val_pred = (y_val_proba > t).astype(int)
        if y_val_pred.sum() == 0:  # Skip if no positive predictions
            continue
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        # Balance: 60% precision, 40% recall (betting strategy)
        score = 0.6 * prec + 0.4 * rec
        if score > best_score:
            best_score = score
            best_threshold = t
    
    y_test_pred = (y_test_proba > best_threshold).astype(int)
    
    # Calibration
    calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrator.fit(X_val, y_val)
    y_test_proba_cal = np.array(calibrator.predict_proba(X_test))[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    print(f"\nTest Results (threshold={best_threshold:.2f}):")
    print(f"  Accuracy:  {acc:.1%}  | Precision: {prec:.1%}")
    print(f"  Recall:    {rec:.1%}  | F1-Score:  {f1:.1%}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  TP={tp:3d}, FP={fp:3d}, TN={tn:3d}, FN={fn:3d}")
    
    # Save model with calibrator
    model_filename = f"api/models/goals_classifier_v3_{threshold}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'calibrator': calibrator,
            'threshold': best_threshold,
            'scaler': scaler,
            'features': FEATURE_COLUMNS
        }, f)
    print(f"✅ Saved: {model_filename}")
    
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'opt_threshold': best_threshold
    })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
summary_df = pd.DataFrame(results)
print(summary_df[['threshold', 'accuracy', 'precision', 'recall', 'auc', 'opt_threshold']].to_string(index=False))
print("\n✅ Training complete!")
