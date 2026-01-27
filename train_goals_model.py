"""
SCOPE - Goals Model Training Script
Training model for Total Goals predictions (FTHG + FTAG)

This script trains a separate model for goals predictions with over/under thresholds.
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - GOALS MODEL
# =============================================================================

LEAGUE_CODE = os.environ.get('LEAGUE_CODE', 'E0')  # EPL default; options: E0, SP1, I1, D1, F1
MODEL_VERSION = "v2_goals_lgbm_tuned"
MODEL_TYPE = "lightgbm"  # "xgboost" or "lightgbm"

ROLLING_WINDOW = 7  # Increased window for better context
TEST_SEASON = '2025-26'
VALIDATION_SPLIT = 0.15  # Reduced to have more training data

LIGHTGBM_PARAMS = {
    'objective': 'poisson',
    'metric': 'rmse',
    'num_leaves': 45,  # Balance between underfitting and overfitting
    'max_depth': 7,
    'learning_rate': 0.03,  # Lower for better generalization
    'n_estimators': 1200,  # More iterations with lower LR
    'min_child_samples': 15,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.05,  # Stronger regularization
    'reg_lambda': 0.15,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

EARLY_STOPPING_ROUNDS = None

# =============================================================================
# DATA LOADING
# =============================================================================
print("="*70)
print(f"SCOPE GOALS MODEL TRAINING - {MODEL_VERSION}")
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
print(f"Goals distribution:")
print(df['TotalGoals'].describe())

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("\nComputing features...")

def compute_rolling_features_goals(df, n=5):
    """Compute venue-aware rolling features for goals prediction."""
    rolling_cols = [
        'home_goals_for', 'home_goals_against', 'home_goals_total', 'home_goals_std',
        'away_goals_for', 'away_goals_against', 'away_goals_total', 'away_goals_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_shot_dominance', 'away_shot_dominance',
        'home_corners_for', 'home_corners_against',
        'away_corners_for', 'away_corners_against',
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
                df.loc[idx, 'home_goals_total'] = (prev_data['FTHG'] + prev_data['FTAG']).mean()
                df.loc[idx, 'home_goals_std'] = prev_data['FTHG'].std()
                df.loc[idx, 'home_shots_for'] = prev_data['HS'].mean()
                df.loc[idx, 'home_shots_against'] = prev_data['AS'].mean()
                df.loc[idx, 'home_sot_for'] = prev_data['HST'].mean()
                df.loc[idx, 'home_sot_against'] = prev_data['AST'].mean()
                df.loc[idx, 'home_shot_dominance'] = prev_data['HS'].mean() - prev_data['AS'].mean()
                df.loc[idx, 'home_corners_for'] = prev_data['HC'].mean()
                df.loc[idx, 'home_corners_against'] = prev_data['AC'].mean()

        # AWAY games
        for i, idx in enumerate(away_indices):
            if i >= n:
                prev = away_indices[i-n:i]
                prev_data = df.loc[prev]

                df.loc[idx, 'away_goals_for'] = prev_data['FTAG'].mean()
                df.loc[idx, 'away_goals_against'] = prev_data['FTHG'].mean()
                df.loc[idx, 'away_goals_total'] = (prev_data['FTHG'] + prev_data['FTAG']).mean()
                df.loc[idx, 'away_goals_std'] = prev_data['FTAG'].std()
                df.loc[idx, 'away_shots_for'] = prev_data['AS'].mean()
                df.loc[idx, 'away_shots_against'] = prev_data['HS'].mean()
                df.loc[idx, 'away_sot_for'] = prev_data['AST'].mean()
                df.loc[idx, 'away_sot_against'] = prev_data['HST'].mean()
                df.loc[idx, 'away_shot_dominance'] = prev_data['AS'].mean() - prev_data['HS'].mean()
                df.loc[idx, 'away_corners_for'] = prev_data['AC'].mean()
                df.loc[idx, 'away_corners_against'] = prev_data['HC'].mean()

    return df

df = compute_rolling_features_goals(df, n=ROLLING_WINDOW)

# =============================================================================
# DERIVED FEATURES
# =============================================================================
print("Computing derived features...")

# Combined metrics
df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']

# Efficiency
df['home_shot_accuracy'] = df['home_sot_for'] / (df['home_shots_for'] + 0.001)
df['away_shot_accuracy'] = df['away_sot_for'] / (df['away_shots_for'] + 0.001)
df['combined_shot_accuracy'] = df['combined_sot_for'] / (df['combined_shots_for'] + 0.001)

# Volatility
df['home_goals_cv'] = df['home_goals_std'] / (df['home_goals_for'] + 0.001)
df['away_goals_cv'] = df['away_goals_std'] / (df['away_goals_for'] + 0.001)
df['combined_goals_volatility'] = df[['home_goals_cv', 'away_goals_cv']].mean(axis=1)

# Goal differential
df['home_goal_balance'] = df['home_goals_for'] - df['home_goals_against']
df['away_goal_balance'] = df['away_goals_for'] - df['away_goals_against']
df['goal_differential'] = df['home_goal_balance'] + df['away_goal_balance']

# Shooting pressure
df['home_shot_share'] = df['home_shots_for'] / (df['combined_shots_for'] + 0.001)
df['away_shot_share'] = df['away_shots_for'] / (df['combined_shots_for'] + 0.001)

# NEW ADVANCED FEATURES
# Conversion rate (goals per shot on target)
df['home_conversion_rate'] = df['home_goals_for'] / (df['home_sot_for'] + 0.001)
df['away_conversion_rate'] = df['away_goals_for'] / (df['away_sot_for'] + 0.001)

# Expected goals (simple xG proxy)
df['home_xg_proxy'] = df['home_sot_for'] * 0.35  # ~35% of SOT = goals
df['away_xg_proxy'] = df['away_sot_for'] * 0.35

# Attack strength vs defense strength
df['attack_strength_diff'] = df['home_goals_for'] - df['away_goals_against']
df['defense_quality_diff'] = df['away_goals_for'] - df['home_goals_against']

# Total attacking intent (shots + corners)
df['home_attacking_intent'] = df['home_shots_for'] + df['home_corners_for'] * 0.5
df['away_attacking_intent'] = df['away_shots_for'] + df['away_corners_for'] * 0.5
df['combined_attacking_intent'] = df['home_attacking_intent'] + df['away_attacking_intent']

# Shot quality (SOT/Shots ratio weighted by goals)
df['home_shot_quality'] = (df['home_sot_for'] / (df['home_shots_for'] + 0.001)) * df['home_goals_for']
df['away_shot_quality'] = (df['away_sot_for'] / (df['away_shots_for'] + 0.001)) * df['away_goals_for']

# Fill remaining NaNs
df = df.fillna(0)

# =============================================================================
# FEATURE SELECTION
# =============================================================================
FEATURE_COLUMNS = [
    # Goals (attacking ability)
    'home_goals_for', 'home_goals_against', 'home_goals_total',
    'away_goals_for', 'away_goals_against', 'away_goals_total',
    'combined_goals_for', 'combined_goals_against',

    # Shots
    'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
    'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
    'combined_shots_for', 'combined_sot_for',

    # Shot Dominance
    'home_shot_dominance', 'away_shot_dominance', 'home_shot_share', 'away_shot_share',

    # Efficiency
    'home_shot_accuracy', 'away_shot_accuracy', 'combined_shot_accuracy',

    # Corners (as indicator of pressure)
    'home_corners_for', 'home_corners_against',
    'away_corners_for', 'away_corners_against',

    # Volatility & Balance
    'home_goals_cv', 'away_goals_cv', 'combined_goals_volatility',
    'home_goal_balance', 'away_goal_balance', 'goal_differential',
    
    # NEW: Advanced features
    'home_conversion_rate', 'away_conversion_rate',
    'home_xg_proxy', 'away_xg_proxy',
    'attack_strength_diff', 'defense_quality_diff',
    'home_attacking_intent', 'away_attacking_intent', 'combined_attacking_intent',
    'home_shot_quality', 'away_shot_quality',
]

TARGET_COLUMN = 'TotalGoals'

print(f"Features: {len(FEATURE_COLUMNS)}")

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
df_model = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
print(f"Matches with complete features: {len(df_model)}")

train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

print(f"Training: {len(train_df)} | Test: {len(test_df)}")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]

# Validation split
val_size = int(len(X_train) * VALIDATION_SPLIT)
X_train_fit = X_train.iloc[:-val_size]
y_train_fit = y_train.iloc[:-val_size]
X_val = X_train.iloc[-val_size:]
y_val = y_train.iloc[-val_size:]

print(f"Train fit: {len(X_train_fit)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "="*70)
print(f"TRAINING ({MODEL_TYPE.upper()})")
print("="*70)

callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if EARLY_STOPPING_ROUNDS else []

model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

if EARLY_STOPPING_ROUNDS:
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Best validation RMSE: {model.best_score_['valid_0']['rmse']:.4f}")
else:
    print(f"Trained for {LIGHTGBM_PARAMS['n_estimators']} iterations")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

y_pred_test = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
corr = np.corrcoef(y_test, y_pred_test)[0, 1]

print(f"\nTest Set Results:")
print(f"  RMSE:        {rmse:.3f}")
print(f"  MAE:         {mae:.3f}")
print(f"  R²:          {r2:.4f}")
print(f"  Correlation: {corr:.4f}")

# Prediction range check
print(f"\n  Pred Range:  {y_pred_test.min():.1f} - {y_pred_test.max():.1f}")
print(f"  Actual Range: {y_test.min():.0f} - {y_test.max():.0f}")

# Over/Under quick check
print("\nOver/Under Accuracy (Total Goals):")
y_test_arr = np.array(y_test)
y_pred_test_arr = np.array(y_pred_test)
for t in [1.5, 2.5, 3.5, 4.5]:
    acc = ((y_test_arr > t) == (y_pred_test_arr > t)).mean() * 100
    print(f"  O/U {t}: {acc:.1f}%")

# =============================================================================
# SAVE MODEL
# =============================================================================
output_dir = os.path.join('api', 'models', LEAGUE_CODE)
os.makedirs(output_dir, exist_ok=True)
model_filename = os.path.join(output_dir, f'model_{LEAGUE_CODE}_{MODEL_VERSION}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
with open(model_filename, 'wb') as f:
    pickle.dump({
        'model': model,
        'model_type': MODEL_TYPE,
        'feature_columns': FEATURE_COLUMNS,
        'params': LIGHTGBM_PARAMS,
        'rolling_window': ROLLING_WINDOW,
        'version': MODEL_VERSION,
        'target': TARGET_COLUMN,
        'test_metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr},
        'train_date': datetime.now().isoformat(),
        'league_code': LEAGUE_CODE
    }, f)

print(f"\nModel saved: {model_filename}")
print("\nGoals model training complete!")
