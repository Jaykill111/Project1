"""
Enhanced Goals Model with External Data Features
New features:
1. Win/Loss streaks (recent momentum)
2. Head-to-head statistics
3. League-wide goal trends
4. Recent goal-scoring streaks
5. Defensive consistency
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

ROLLING_WINDOW = 7
TEST_SEASON = '2025-26'
THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

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
# LOAD DATA
# =============================================================================
print("="*70)
print("GOALS MODEL WITH EXTERNAL DATA FEATURES")
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
# BASE FEATURES (same as V3)
# =============================================================================
print("Computing base features...")

def compute_rolling_features_goals(df, n=7):
    """Compute rolling features for goals prediction."""
    rolling_cols = [
        'home_goals_for', 'home_goals_against', 'home_goals_std',
        'away_goals_for', 'away_goals_against', 'away_goals_std',
        'home_shots_for', 'home_shots_against', 'home_sot_for', 'home_sot_against',
        'away_shots_for', 'away_shots_against', 'away_sot_for', 'away_sot_against',
        'home_corners_for', 'away_corners_for',
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
                    df.loc[idx, 'home_goals_for_3'] = df.loc[prev_3, 'FTHG'].mean()

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
                    df.loc[idx, 'away_goals_for_3'] = df.loc[prev_3, 'FTAG'].mean()

    return df

df = compute_rolling_features_goals(df, n=ROLLING_WINDOW)

# Base derived features
df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
df['home_goal_balance'] = df['home_goals_for'] - df['home_goals_against']
df['away_goal_balance'] = df['away_goals_for'] - df['away_goals_against']

df['home_shot_accuracy'] = df['home_sot_for'] / (df['home_shots_for'] + 0.001)
df['away_shot_accuracy'] = df['away_sot_for'] / (df['away_shots_for'] + 0.001)
df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']

df['home_conversion_rate'] = df['home_goals_for'] / (df['home_sot_for'] + 0.001)
df['away_conversion_rate'] = df['away_goals_for'] / (df['away_sot_for'] + 0.001)

df['home_goals_cv'] = df['home_goals_std'] / (df['home_goals_for'] + 0.001)
df['away_goals_cv'] = df['away_goals_std'] / (df['away_goals_for'] + 0.001)

df['home_momentum'] = df['home_goals_for_3'] - df['home_goals_for']
df['away_momentum'] = df['away_goals_for_3'] - df['away_goals_for']
df['combined_momentum'] = df['home_momentum'] + df['away_momentum']

df['attack_diff'] = df['home_goals_for'] - df['away_goals_against']
df['defense_diff'] = df['away_goals_for'] - df['home_goals_against']

# =============================================================================
# NEW EXTERNAL FEATURES
# =============================================================================
print("\nComputing external features...")

# 1. WIN/LOSS STREAKS
def compute_win_streaks(df, n=7):
    """Win/loss streaks in last n games."""
    df['home_wins'] = np.nan
    df['home_losses'] = np.nan
    df['away_wins'] = np.nan
    df['away_losses'] = np.nan
    
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in all_teams:
        home_idx = df[df['HomeTeam'] == team].index.tolist()
        away_idx = df[df['AwayTeam'] == team].index.tolist()
        
        for i, idx in enumerate(home_idx):
            if i >= n:
                prev = home_idx[i-n:i]
                prev_data = df.loc[prev]
                wins = ((prev_data['FTHG'] > prev_data['FTAG']).sum())
                losses = ((prev_data['FTHG'] < prev_data['FTAG']).sum())
                df.loc[idx, 'home_wins'] = wins
                df.loc[idx, 'home_losses'] = losses
        
        for i, idx in enumerate(away_idx):
            if i >= n:
                prev = away_idx[i-n:i]
                prev_data = df.loc[prev]
                wins = ((prev_data['FTAG'] > prev_data['FTHG']).sum())
                losses = ((prev_data['FTAG'] < prev_data['FTHG']).sum())
                df.loc[idx, 'away_wins'] = wins
                df.loc[idx, 'away_losses'] = losses
    
    return df

df = compute_win_streaks(df, n=ROLLING_WINDOW)

# Win rate
df['home_win_rate'] = df['home_wins'] / (df['home_wins'] + df['home_losses'] + 0.001)
df['away_win_rate'] = df['away_wins'] / (df['away_wins'] + df['away_losses'] + 0.001)

# 2. RECENT GOAL-SCORING STREAKS (high-scoring games in last 5 matches)
def compute_goal_streaks(df, threshold=2.5, n=5):
    """Count matches with >threshold goals in last n games."""
    df['home_high_scoring_streak'] = np.nan
    df['away_high_scoring_streak'] = np.nan
    
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in all_teams:
        home_idx = df[df['HomeTeam'] == team].index.tolist()
        away_idx = df[df['AwayTeam'] == team].index.tolist()
        
        for i, idx in enumerate(home_idx):
            if i >= n:
                prev = home_idx[i-n:i]
                prev_data = df.loc[prev]
                total_goals = (prev_data['FTHG'] + prev_data['FTAG']).values
                high_scoring = (total_goals > threshold).sum()
                df.loc[idx, 'home_high_scoring_streak'] = high_scoring / n
        
        for i, idx in enumerate(away_idx):
            if i >= n:
                prev = away_idx[i-n:i]
                prev_data = df.loc[prev]
                total_goals = (prev_data['FTHG'] + prev_data['FTAG']).values
                high_scoring = (total_goals > threshold).sum()
                df.loc[idx, 'away_high_scoring_streak'] = high_scoring / n
    
    return df

df = compute_goal_streaks(df, threshold=2.5, n=5)

# 3. HEAD-TO-HEAD STATISTICS
def compute_h2h_stats(df, min_matches=3):
    """Historical O/U record between teams."""
    df['h2h_over_pct'] = np.nan
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        current_date = row['Date']
        
        # Find previous matches between these teams
        h2h = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) |
             (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
            (df['Date'] < current_date)
        ]
        
        if len(h2h) >= min_matches:
            over_pct = ((h2h['FTHG'] + h2h['FTAG']) > 2.5).mean()
            df.loc[idx, 'h2h_over_pct'] = over_pct
        else:
            df.loc[idx, 'h2h_over_pct'] = 0.5  # Default to 50%
    
    return df

print("  Computing head-to-head stats... (this may take a moment)")
df = compute_h2h_stats(df)

# 4. LEAGUE-WIDE TREND (rolling mean of goals across all matches)
def compute_league_trend(df, window=10):
    """League-wide average goals per match (trend)."""
    league_avg = df['TotalGoals'].rolling(window=window, min_periods=1).mean()
    df['league_goal_trend'] = league_avg
    return df

df = compute_league_trend(df, window=10)

# 5. DEFENSIVE SOLIDITY
def compute_defensive_metrics(df, n=7):
    """Shots conceded per match, consistency in defense."""
    df['home_shots_conceded_avg'] = np.nan
    df['away_shots_conceded_avg'] = np.nan
    df['home_defensive_variance'] = np.nan
    df['away_defensive_variance'] = np.nan
    
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in all_teams:
        home_idx = df[df['HomeTeam'] == team].index.tolist()
        away_idx = df[df['AwayTeam'] == team].index.tolist()
        
        for i, idx in enumerate(home_idx):
            if i >= n:
                prev = home_idx[i-n:i]
                prev_data = df.loc[prev]
                df.loc[idx, 'home_shots_conceded_avg'] = prev_data['AS'].mean()
                df.loc[idx, 'home_defensive_variance'] = prev_data['FTAG'].std()
        
        for i, idx in enumerate(away_idx):
            if i >= n:
                prev = away_idx[i-n:i]
                prev_data = df.loc[prev]
                df.loc[idx, 'away_shots_conceded_avg'] = prev_data['HS'].mean()
                df.loc[idx, 'away_defensive_variance'] = prev_data['FTHG'].std()
    
    return df

df = compute_defensive_metrics(df, n=ROLLING_WINDOW)

df = df.fillna(0)

# =============================================================================
# FEATURE SELECTION (WITH NEW EXTERNAL FEATURES)
# =============================================================================
FEATURE_COLUMNS = [
    # Base features
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
    # NEW: External features
    'home_wins', 'home_losses', 'away_wins', 'away_losses',
    'home_win_rate', 'away_win_rate',
    'home_high_scoring_streak', 'away_high_scoring_streak',
    'h2h_over_pct',
    'league_goal_trend',
    'home_shots_conceded_avg', 'away_shots_conceded_avg',
    'home_defensive_variance', 'away_defensive_variance',
]

print(f"Total features: {len(FEATURE_COLUMNS)} (base: 35 + external: {len(FEATURE_COLUMNS)-35})")

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

# Normalize
for feat in FEATURE_COLUMNS:
    p95 = X_train[feat].quantile(0.95)
    p5 = X_train[feat].quantile(0.05)
    X_train[feat] = X_train[feat].clip(p5, p95)
    X_test[feat] = X_test[feat].clip(p5, p95)

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
    
    # SMOTE
    if HAS_SMOTE and threshold in [2.5, 3.5]:
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_fit_smote, y_train_fit_smote = smote.fit_resample(X_train_fit, y_train_fit)
            X_train_fit = pd.DataFrame(X_train_fit_smote, columns=FEATURE_COLUMNS)
            y_train_fit = pd.Series(y_train_fit_smote).reset_index(drop=True)
        except Exception as e:
            pass
    
    # Train
    params = LIGHTGBM_PARAMS.copy()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    y_val_proba = np.array(model.predict_proba(X_val))[:, 1]
    y_test_proba = np.array(model.predict_proba(X_test))[:, 1]
    
    # Optimize threshold
    best_threshold = 0.5
    best_score = 0
    for t in np.arange(0.2, 0.8, 0.05):
        y_val_pred = (y_val_proba > t).astype(int)
        if y_val_pred.sum() == 0:
            continue
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        score = 0.6 * prec + 0.4 * rec
        if score > best_score:
            best_score = score
            best_threshold = t
    
    y_test_pred = (y_test_proba > best_threshold).astype(int)
    
    # Calibrate
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
    
    # Save
    model_filename = f"api/models/goals_classifier_external_{threshold}.pkl"
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
    })

# Summary
print("\n" + "="*70)
print("COMPARISON: Base (V3) vs External Features")
print("="*70)
summary_df = pd.DataFrame(results)
print(summary_df[['threshold', 'accuracy', 'precision', 'recall', 'auc']].to_string(index=False))

v3_results = [
    {'threshold': 1.5, 'auc': 0.522},
    {'threshold': 2.5, 'auc': 0.509},
    {'threshold': 3.5, 'auc': 0.484},
    {'threshold': 4.5, 'auc': 0.562},
]
v3_df = pd.DataFrame(v3_results)

print("\nAUC-ROC Improvement:")
for _, row in summary_df.iterrows():
    t = row['threshold']
    new_auc = row['auc']
    old_auc = v3_df[v3_df['threshold'] == t]['auc'].values[0]
    improvement = (new_auc - old_auc) / old_auc * 100
    symbol = "✓" if improvement > 0 else "✗"
    print(f"  {symbol} {t}: {old_auc:.3f} → {new_auc:.3f} ({improvement:+.1f}%)")

print("\n✅ Training complete!")
