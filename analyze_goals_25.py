"""
Analyze data quality issues for Goals Model - Threshold 2.5
Identify class imbalance, outliers, and feature correlation problems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DATA QUALITY ANALYSIS: GOALS MODEL - THRESHOLD 2.5")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
ROLLING_WINDOW = 7
TEST_SEASON = '2025-26'
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

# ============================================================================
# ANALYSIS 1: CLASS IMBALANCE FOR 2.5 THRESHOLD
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: CLASS IMBALANCE")
print("="*70)

over_25 = (df['TotalGoals'] > 2.5).sum()
under_25 = (df['TotalGoals'] <= 2.5).sum()
total = len(df)

print(f"\nTotalGoals > 2.5 (OVER):  {over_25:4d} ({over_25/total*100:.1f}%)")
print(f"TotalGoals ≤ 2.5 (UNDER): {under_25:4d} ({under_25/total*100:.1f}%)")
print(f"Total: {total}")

# By season
print("\n📊 Distribution by season:")
for season in sorted(df['Season'].unique()):
    season_df = df[df['Season'] == season]
    over = (season_df['TotalGoals'] > 2.5).sum()
    total_s = len(season_df)
    pct = over / total_s * 100 if total_s > 0 else 0
    print(f"  {season}: Over={over:3d}/{total_s:3d} ({pct:.1f}%)")

# ============================================================================
# ANALYSIS 2: FEATURE QUALITY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: FEATURE QUALITY & MISSING VALUES")
print("="*70)

# Compute rolling features
def compute_rolling_features_goals(df, n=7):
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
df['combined_goals_for'] = df['home_goals_for'] + df['away_goals_for']
df['combined_goals_against'] = df['home_goals_against'] + df['away_goals_against']
df['home_goal_balance'] = df['home_goals_for'] - df['home_goals_against']
df['away_goal_balance'] = df['away_goals_for'] - df['away_goals_against']
df['goal_differential'] = df['home_goal_balance'] + df['away_goal_balance']

df['home_shot_accuracy'] = df['home_sot_for'] / (df['home_shots_for'] + 0.001)
df['away_shot_accuracy'] = df['away_sot_for'] / (df['away_shots_for'] + 0.001)
df['combined_shots_for'] = df['home_shots_for'] + df['away_shots_for']
df['combined_sot_for'] = df['home_sot_for'] + df['away_sot_for']

df['home_conversion_rate'] = df['home_goals_for'] / (df['home_sot_for'] + 0.001)
df['away_conversion_rate'] = df['away_goals_for'] / (df['away_sot_for'] + 0.001)

df['home_xg_proxy'] = df['home_sot_for'] * 0.35
df['away_xg_proxy'] = df['away_sot_for'] * 0.35
df['combined_xg_proxy'] = df['home_xg_proxy'] + df['away_xg_proxy']

df['attack_strength_diff'] = df['home_goals_for'] - df['away_goals_against']
df['defense_quality_diff'] = df['away_goals_for'] - df['home_goals_against']

df['home_goals_cv'] = df['home_goals_std'] / (df['home_goals_for'] + 0.001)
df['away_goals_cv'] = df['away_goals_std'] / (df['away_goals_for'] + 0.001)

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

df_model = df.dropna(subset=FEATURE_COLUMNS + ['TotalGoals']).copy()

print(f"\nRows after feature computation: {len(df_model)}/{len(df)} ({len(df_model)/len(df)*100:.1f}%)")

# Missing values per feature
print("\n📋 Missing values:")
missing_counts = df[FEATURE_COLUMNS].isnull().sum()
missing_pcts = missing_counts / len(df) * 100
for feat, count, pct in zip(missing_counts.index, missing_counts, missing_pcts):
    if pct > 0:
        print(f"  {feat:30s}: {count:4d} ({pct:.1f}%)")

# ============================================================================
# ANALYSIS 3: FEATURE CORRELATION WITH TARGET
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: FEATURE CORRELATION WITH TARGET")
print("="*70)

df_model['target'] = (df_model['TotalGoals'] > 2.5).astype(int)

correlations = {}
for feat in FEATURE_COLUMNS:
    corr = df_model[feat].corr(df_model['target'])
    correlations[feat] = corr

# Sort by absolute correlation
sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 10 most predictive features:")
for i, (feat, corr) in enumerate(sorted_corrs[:10], 1):
    print(f"  {i:2d}. {feat:30s}: {corr:+.4f}")

print("\nBottom 5 least predictive features:")
for i, (feat, corr) in enumerate(sorted_corrs[-5:], 1):
    print(f"  {i:2d}. {feat:30s}: {corr:+.4f}")

# ============================================================================
# ANALYSIS 4: OUTLIERS & EXTREME VALUES
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 4: OUTLIERS & EXTREME VALUES")
print("="*70)

print("\nFeature statistics (for training data):")
train_df = df_model[df_model['Season'] != TEST_SEASON]
print("\nTop goal scorers in training:")
for feat in ['home_goals_for', 'away_goals_for', 'combined_goals_for']:
    print(f"  {feat}:")
    print(f"    Min: {train_df[feat].min():.2f}, Max: {train_df[feat].max():.2f}")
    print(f"    Mean: {train_df[feat].mean():.2f}, Std: {train_df[feat].std():.2f}")
    print(f"    75th %ile: {train_df[feat].quantile(0.75):.2f}, 95th %ile: {train_df[feat].quantile(0.95):.2f}")

# Outlier detection via IQR
print("\n🚨 Outliers detected (IQR method):")
outlier_count = 0
for feat in ['home_goals_for', 'away_goals_for', 'combined_goals_for']:
    Q1 = train_df[feat].quantile(0.25)
    Q3 = train_df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((train_df[feat] < lower) | (train_df[feat] > upper)).sum()
    print(f"  {feat:30s}: {outliers:3d} outliers ({outliers/len(train_df)*100:.1f}%)")
    outlier_count += outliers

# ============================================================================
# ANALYSIS 5: TEMPORAL STABILITY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 5: TEMPORAL STABILITY")
print("="*70)

print("\nGoal statistics by season:")
for season in sorted(df_model['Season'].unique()):
    season_df = df_model[df_model['Season'] == season]
    mean_goals = season_df['TotalGoals'].mean()
    std_goals = season_df['TotalGoals'].std()
    over_pct = (season_df['TotalGoals'] > 2.5).mean() * 100
    print(f"  {season}: μ={mean_goals:.2f}, σ={std_goals:.2f}, Over%={over_pct:.1f}%")

# ============================================================================
# ANALYSIS 6: FEATURE REDUNDANCY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 6: MULTICOLLINEARITY CHECK")
print("="*70)

# Check correlation among features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS])
corr_matrix = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS).corr()

# Find highly correlated pairs
print("\n⚠️  Highly correlated feature pairs (|r| > 0.9):")
high_corr_pairs = 0
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            print(f"  {corr_matrix.columns[i]:30s} <-> {corr_matrix.columns[j]:30s}: {corr_matrix.iloc[i, j]:+.4f}")
            high_corr_pairs += 1

if high_corr_pairs == 0:
    print("  (None found - multicollinearity is reasonable)")

print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

imbalance_ratio = under_25 / over_25 if over_25 > 0 else 0
print(f"""
🔍 KEY FINDINGS:

1. CLASS IMBALANCE: {imbalance_ratio:.2f}:1 (UNDER:OVER)
   → Recommendation: Use SMOTE oversampling or class weights

2. LOW FEATURE CORRELATION: Best feature has corr = {sorted_corrs[0][1]:.4f}
   → Recommendation: Engineer new features (e.g., recent trend, volatility)

3. MISSING DATA: {missing_counts.sum()} total NaN values removed
   → Recommendation: Use rolling window > {ROLLING_WINDOW} or forward fill

4. OUTLIERS: ~{outlier_count} detected in key features
   → Recommendation: Cap at 95th percentile or use robust scaling

5. FEATURES: {len(FEATURE_COLUMNS)} total features
   → Recommendation: Feature selection to reduce noise
""")

print("="*70)
