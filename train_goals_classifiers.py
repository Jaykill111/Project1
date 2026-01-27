"""
SCOPE - Goals Classifiers Training Script
Train binary classifiers for each O/U threshold (1.5, 2.5, 3.5, 4.5)

This approach is better than regression because:
- Direct probability outputs for betting
- Better for imbalanced classes
- Matches corners model architecture
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

LEAGUE_CODE = os.environ.get('LEAGUE_CODE', 'E0')  # EPL default; options: E0, SP1, I1, D1, F1
ROLLING_WINDOW = 7
TEST_SEASON = '2025-26'
THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

# LightGBM parameters for classification
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

EARLY_STOPPING_ROUNDS = 50

# =============================================================================
# DATA LOADING
# =============================================================================
print("="*70)
print("SCOPE GOALS CLASSIFIERS TRAINING")
print("="*70)

SEASONS = {
    '2020-21': '2021',
    '2021-22': '2122',
    '2022-23': '2223',
    '2023-24': '2324',
    '2024-25': '2425',
    '2025-26': '2526'
}

BASE_URL = f'https://www.football-data.co.uk/mmz4281' + '/{code}/' + f'{LEAGUE_CODE}.csv'
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
# FEATURE ENGINEERING
# =============================================================================
print("\nComputing features...")

def compute_rolling_features_goals(df, n=7):
    """Compute venue-aware rolling features for goals prediction."""
    rolling_cols = [
        'home_goals_for', 'home_goals_against', 'home_goals_std',
        'away_goals_for', 'away_goals_against', 'away_goals_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_corners_for', 'away_corners_for',
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

    return df

df = compute_rolling_features_goals(df, n=ROLLING_WINDOW)

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

# xG proxy
df['home_xg_proxy'] = df['home_sot_for'] * 0.35
df['away_xg_proxy'] = df['away_sot_for'] * 0.35
df['combined_xg_proxy'] = df['home_xg_proxy'] + df['away_xg_proxy']

# Attack vs defense
df['attack_strength_diff'] = df['home_goals_for'] - df['away_goals_against']
df['defense_quality_diff'] = df['away_goals_for'] - df['home_goals_against']

# Volatility
df['home_goals_cv'] = df['home_goals_std'] / (df['home_goals_for'] + 0.001)
df['away_goals_cv'] = df['away_goals_std'] / (df['away_goals_for'] + 0.001)

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
    'home_goal_balance', 'away_goal_balance', 'goal_differential',
    'home_shot_accuracy', 'away_shot_accuracy',
    'combined_shots_for', 'combined_sot_for',
    'home_conversion_rate', 'away_conversion_rate',
    'home_xg_proxy', 'away_xg_proxy', 'combined_xg_proxy',
    'attack_strength_diff', 'defense_quality_diff',
    'home_goals_cv', 'away_goals_cv',
]

print(f"Features: {len(FEATURE_COLUMNS)}")

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
    print(f"\nClass distribution:")
    print(f"  Train - Over: {train_over_pct:.1f}% | Under: {100-train_over_pct:.1f}%")
    print(f"  Test  - Over: {test_over_pct:.1f}% | Under: {100-test_over_pct:.1f}%")
    
    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train/val split (temporal)
    val_size = int(len(X_train) * 0.15)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    
    # Adjust params for this threshold
    params = LIGHTGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    # Predictions with probability
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold using validation set
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
    
    # Use optimal threshold for test predictions
    y_pred = (y_pred_proba > best_threshold).astype(int)
    
    # Optional calibration (Isotonic)
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
    if calibrator is not None:
        try:
            auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
            print(f"  AUC-ROC (calibrated):   {auc_cal:.3f}")
        except Exception:
            pass
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Best iter: {model.best_iteration_}")
    
    # Save model
    output_dir = os.path.join('api', 'models', LEAGUE_CODE)
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, f'goals_{LEAGUE_CODE}_ou{threshold}_w{ROLLING_WINDOW}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    
    # Encode team names for prediction compatibility
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_home.fit(df_model['HomeTeam'])
    le_away.fit(df_model['AwayTeam'])
    
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': threshold,
            'features': FEATURE_COLUMNS,
            'window': ROLLING_WINDOW,
            'le_home': le_home,
            'le_away': le_away,
            'optimal_threshold': best_threshold,
            'calibrator': calibrator,
            'league_code': LEAGUE_CODE,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            },
            'train_date': datetime.now().isoformat(),
            'model_type': 'goals_classifier'
        }, f)
    
    print(f"\nModel saved: {model_filename}")
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'filename': model_filename
    })

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("TRAINING SUMMARY - GOALS CLASSIFIERS")
print("="*70)

print(f"\n{'Threshold':<12} {'Accuracy':<12} {'AUC':<12} {'F1':<12}")
print("-" * 50)
for r in results:
    print(f"O/U {r['threshold']:<6} {r['accuracy']:<12.1%} {r['auc']:<12.3f} {r['f1']:<12.3f}")

print("\n✓ All classifiers trained successfully!")
print(f"✓ Models saved to: api/models/")
