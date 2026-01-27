"""
SCOPE - Feature Engineering Module V2
New features: shot accuracy, corners per shot, yellow cards, goal difference
"""
import pandas as pd
import numpy as np

# Feature list used by the models (V2)
FEATURES = [
    'home_corners_avg', 'away_corners_avg',
    'home_corners_conceded', 'away_corners_conceded',
    'home_shots_avg', 'away_shots_avg',
    'home_sot_avg', 'away_sot_avg',
    'home_shot_accuracy', 'away_shot_accuracy',
    'home_corners_per_shot', 'away_corners_per_shot',
    'home_yellows_avg', 'away_yellows_avg',
    'home_goal_diff', 'away_goal_diff',
    'home_fouls_avg', 'away_fouls_avg',
    'total_corners_expected', 'total_shots_expected',
    'corner_efficiency_combined', 'aggression_combined', 'form_diff',
]


def compute_features(df, n=5):
    """
    Compute rolling window features for corner prediction (V2).

    Args:
        df: DataFrame with columns Date, HomeTeam, AwayTeam, HC, AC, HS, AS, HST, AST, FTHG, FTAG, HF, AF, HY, AY
        n: Rolling window size (number of previous home/away games)

    Returns:
        DataFrame with computed features
    """
    feature_cols = [
        # Core corner stats
        'home_corners_avg', 'away_corners_avg',
        'home_corners_conceded', 'away_corners_conceded',
        # Shot stats
        'home_shots_avg', 'away_shots_avg',
        'home_sot_avg', 'away_sot_avg',
        # Shot efficiency (SOT/Shots)
        'home_shot_accuracy', 'away_shot_accuracy',
        # Corners per shot (corner generation efficiency)
        'home_corners_per_shot', 'away_corners_per_shot',
        # Yellow cards (aggression indicator)
        'home_yellows_avg', 'away_yellows_avg',
        # Goal difference (form indicator)
        'home_goal_diff', 'away_goal_diff',
        # Fouls
        'home_fouls_avg', 'away_fouls_avg',
    ]
    for col in feature_cols:
        df[col] = np.nan

    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for team in all_teams:
        home_idx = df[df['HomeTeam'] == team].index.tolist()
        away_idx = df[df['AwayTeam'] == team].index.tolist()

        for i, idx in enumerate(home_idx):
            if i >= n:
                prev = df.loc[home_idx[i-n:i]]
                df.loc[idx, 'home_corners_avg'] = prev['HC'].mean()
                df.loc[idx, 'home_corners_conceded'] = prev['AC'].mean()
                df.loc[idx, 'home_shots_avg'] = prev['HS'].mean()
                df.loc[idx, 'home_sot_avg'] = prev['HST'].mean()

                # Shot accuracy
                shots = prev['HS'].sum()
                if shots > 0:
                    df.loc[idx, 'home_shot_accuracy'] = prev['HST'].sum() / shots
                    df.loc[idx, 'home_corners_per_shot'] = prev['HC'].sum() / shots

                # Yellow cards
                if 'HY' in prev.columns:
                    df.loc[idx, 'home_yellows_avg'] = prev['HY'].mean()

                # Goal difference
                df.loc[idx, 'home_goal_diff'] = (prev['FTHG'] - prev['FTAG']).mean()

                # Fouls
                if 'HF' in prev.columns:
                    df.loc[idx, 'home_fouls_avg'] = prev['HF'].mean()

        for i, idx in enumerate(away_idx):
            if i >= n:
                prev = df.loc[away_idx[i-n:i]]
                df.loc[idx, 'away_corners_avg'] = prev['AC'].mean()
                df.loc[idx, 'away_corners_conceded'] = prev['HC'].mean()
                df.loc[idx, 'away_shots_avg'] = prev['AS'].mean()
                df.loc[idx, 'away_sot_avg'] = prev['AST'].mean()

                # Shot accuracy
                shots = prev['AS'].sum()
                if shots > 0:
                    df.loc[idx, 'away_shot_accuracy'] = prev['AST'].sum() / shots
                    df.loc[idx, 'away_corners_per_shot'] = prev['AC'].sum() / shots

                # Yellow cards
                if 'AY' in prev.columns:
                    df.loc[idx, 'away_yellows_avg'] = prev['AY'].mean()

                # Goal difference
                df.loc[idx, 'away_goal_diff'] = (prev['FTAG'] - prev['FTHG']).mean()

                # Fouls
                if 'AF' in prev.columns:
                    df.loc[idx, 'away_fouls_avg'] = prev['AF'].mean()

    # Combined features (non-redundant)
    df['total_corners_expected'] = df['home_corners_avg'] + df['away_corners_avg']
    df['total_shots_expected'] = df['home_shots_avg'] + df['away_shots_avg']
    df['corner_efficiency_combined'] = df['home_corners_per_shot'].fillna(0) + df['away_corners_per_shot'].fillna(0)
    df['aggression_combined'] = df['home_yellows_avg'].fillna(0) + df['away_yellows_avg'].fillna(0)
    df['form_diff'] = df['home_goal_diff'].fillna(0) - df['away_goal_diff'].fillna(0)
    return df


def compute_features_for_match(df, home_team, away_team, window=5):
    """
    Compute features for a specific upcoming match (V2).

    Args:
        df: Full DataFrame with historical data
        home_team: Home team name
        away_team: Away team name
        window: Rolling window size

    Returns:
        dict: Feature values for the match
    """
    # Get last N home games for home team
    home_games = df[df['HomeTeam'] == home_team].tail(window)
    # Get last N away games for away team
    away_games = df[df['AwayTeam'] == away_team].tail(window)

    features = {}

    if len(home_games) >= window:
        features['home_corners_avg'] = home_games['HC'].mean()
        features['home_corners_conceded'] = home_games['AC'].mean()
        features['home_shots_avg'] = home_games['HS'].mean() if 'HS' in df.columns else 12.0
        features['home_sot_avg'] = home_games['HST'].mean() if 'HST' in df.columns else 4.0

        # Shot accuracy
        shots = home_games['HS'].sum() if 'HS' in df.columns else 0
        if shots > 0:
            features['home_shot_accuracy'] = home_games['HST'].sum() / shots
            features['home_corners_per_shot'] = home_games['HC'].sum() / shots
        else:
            features['home_shot_accuracy'] = 0.33
            features['home_corners_per_shot'] = 0.4

        # Yellow cards
        features['home_yellows_avg'] = home_games['HY'].mean() if 'HY' in df.columns else 1.5

        # Goal difference
        features['home_goal_diff'] = (home_games['FTHG'] - home_games['FTAG']).mean() if 'FTHG' in df.columns else 0

        # Fouls
        features['home_fouls_avg'] = home_games['HF'].mean() if 'HF' in df.columns else 10.0
    else:
        # Default values if not enough data
        features['home_corners_avg'] = 5.0
        features['home_corners_conceded'] = 5.0
        features['home_shots_avg'] = 12.0
        features['home_sot_avg'] = 4.0
        features['home_shot_accuracy'] = 0.33
        features['home_corners_per_shot'] = 0.4
        features['home_yellows_avg'] = 1.5
        features['home_goal_diff'] = 0
        features['home_fouls_avg'] = 10.0

    if len(away_games) >= window:
        features['away_corners_avg'] = away_games['AC'].mean()
        features['away_corners_conceded'] = away_games['HC'].mean()
        features['away_shots_avg'] = away_games['AS'].mean() if 'AS' in df.columns else 10.0
        features['away_sot_avg'] = away_games['AST'].mean() if 'AST' in df.columns else 3.0

        # Shot accuracy
        shots = away_games['AS'].sum() if 'AS' in df.columns else 0
        if shots > 0:
            features['away_shot_accuracy'] = away_games['AST'].sum() / shots
            features['away_corners_per_shot'] = away_games['AC'].sum() / shots
        else:
            features['away_shot_accuracy'] = 0.30
            features['away_corners_per_shot'] = 0.35

        # Yellow cards
        features['away_yellows_avg'] = away_games['AY'].mean() if 'AY' in df.columns else 1.8

        # Goal difference
        features['away_goal_diff'] = (away_games['FTAG'] - away_games['FTHG']).mean() if 'FTAG' in df.columns else 0

        # Fouls
        features['away_fouls_avg'] = away_games['AF'].mean() if 'AF' in df.columns else 11.0
    else:
        # Default values if not enough data
        features['away_corners_avg'] = 4.0
        features['away_corners_conceded'] = 5.0
        features['away_shots_avg'] = 10.0
        features['away_sot_avg'] = 3.0
        features['away_shot_accuracy'] = 0.30
        features['away_corners_per_shot'] = 0.35
        features['away_yellows_avg'] = 1.8
        features['away_goal_diff'] = 0
        features['away_fouls_avg'] = 11.0

    # Combined features
    features['total_corners_expected'] = features['home_corners_avg'] + features['away_corners_avg']
    features['total_shots_expected'] = features['home_shots_avg'] + features['away_shots_avg']
    features['corner_efficiency_combined'] = features['home_corners_per_shot'] + features['away_corners_per_shot']
    features['aggression_combined'] = features['home_yellows_avg'] + features['away_yellows_avg']
    features['form_diff'] = features['home_goal_diff'] - features['away_goal_diff']

    return features


def get_team_statistics(df, team, n=5):
    """
    Get recent statistics for a team.

    Args:
        df: DataFrame with match data
        team: Team name
        n: Number of recent matches

    Returns:
        dict: Team statistics
    """
    # Home matches
    home = df[df['HomeTeam'] == team].tail(n)
    # Away matches
    away = df[df['AwayTeam'] == team].tail(n)

    # All recent matches (sorted by date)
    all_matches = pd.concat([
        home[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HS', 'AS']].assign(venue='Home'),
        away[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC', 'HS', 'AS']].assign(venue='Away')
    ]).sort_values('Date').tail(n)

    # Calculate corners for/against
    corners_for = []
    corners_against = []
    for _, row in all_matches.iterrows():
        if row['venue'] == 'Home':
            corners_for.append(row['HC'])
            corners_against.append(row['AC'])
        else:
            corners_for.append(row['AC'])
            corners_against.append(row['HC'])

    # Recent form - last 5 matches
    recent_matches = []
    for _, row in all_matches.iterrows():
        if row['venue'] == 'Home':
            gf, ga = row['FTHG'], row['FTAG']
            opponent = row['AwayTeam']
            cf, ca = row['HC'], row['AC']
        else:
            gf, ga = row['FTAG'], row['FTHG']
            opponent = row['HomeTeam']
            cf, ca = row['AC'], row['HC']

        result = 'W' if gf > ga else ('D' if gf == ga else 'L')
        recent_matches.append({
            'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'opponent': opponent,
            'venue': row['venue'],
            'result': result,
            'score': f"{int(gf)}-{int(ga)}",
            'corners_for': int(cf),
            'corners_against': int(ca),
            'total_corners': int(cf + ca)
        })

    # Calculate over rates
    total_corners = [m['total_corners'] for m in recent_matches]

    return {
        'recent_matches': recent_matches,
        'avg_corners_for': np.mean(corners_for) if corners_for else 0,
        'avg_corners_against': np.mean(corners_against) if corners_against else 0,
        'avg_total_corners': np.mean(total_corners) if total_corners else 0,
        'over_rates': {
            '8.5': sum(1 for c in total_corners if c > 8.5) / len(total_corners) if total_corners else 0,
            '9.5': sum(1 for c in total_corners if c > 9.5) / len(total_corners) if total_corners else 0,
            '10.5': sum(1 for c in total_corners if c > 10.5) / len(total_corners) if total_corners else 0,
            '11.5': sum(1 for c in total_corners if c > 11.5) / len(total_corners) if total_corners else 0,
            '12.5': sum(1 for c in total_corners if c > 12.5) / len(total_corners) if total_corners else 0,
        }
    }


def get_head_to_head(df, home_team, away_team, n=5):
    """
    Get head-to-head history between two teams.

    Args:
        df: DataFrame with match data
        home_team: Home team name
        away_team: Away team name
        n: Number of recent H2H matches

    Returns:
        dict: H2H statistics
    """
    # Get matches between these two teams
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].tail(n)

    if len(h2h) == 0:
        return {
            'matches': [],
            'avg_total_corners': 0,
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0
        }

    matches = []
    home_wins = 0
    away_wins = 0
    draws = 0
    total_corners = []

    for _, row in h2h.iterrows():
        total = row['HC'] + row['AC']
        total_corners.append(total)

        if row['FTHG'] > row['FTAG']:
            result = row['HomeTeam']
            if row['HomeTeam'] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        elif row['FTHG'] < row['FTAG']:
            result = row['AwayTeam']
            if row['AwayTeam'] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        else:
            result = 'Draw'
            draws += 1

        matches.append({
            'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'home': row['HomeTeam'],
            'away': row['AwayTeam'],
            'score': f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            'corners': f"{int(row['HC'])}-{int(row['AC'])}",
            'total_corners': int(total),
            'winner': result
        })

    return {
        'matches': matches,
        'avg_total_corners': np.mean(total_corners),
        'home_team_wins': home_wins,
        'away_team_wins': away_wins,
        'draws': draws
    }

def compute_features_for_goals_match(df, home_team, away_team, window=5):
    """
    Compute features for goals prediction for a specific upcoming match.
    Includes both base features and external features matching the trained model.
    
    Args:
        df: Full DataFrame with historical data
        home_team: Home team name
        away_team: Away team name
        window: Rolling window size
    
    Returns:
        dict: Feature values for the match (goals-specific)
    """
    # Get all historical games for each team (not just window)
    home_all_games = df[df['HomeTeam'] == home_team].sort_values('Date')
    away_all_games = df[df['AwayTeam'] == away_team].sort_values('Date')
    
    # Last N games
    home_games = home_all_games.tail(window)
    away_games = away_all_games.tail(window)
    
    # Last 3 games for momentum
    home_games_3 = home_all_games.tail(3)
    away_games_3 = away_all_games.tail(3)

    features = {}

    # =========================================================================
    # BASE FEATURES
    # =========================================================================
    
    # HOME TEAM FEATURES
    if len(home_games) >= window:
        # Goals
        features['home_goals_for'] = home_games['FTHG'].mean() if 'FTHG' in df.columns else 1.5
        features['home_goals_against'] = home_games['FTAG'].mean() if 'FTAG' in df.columns else 1.2
        features['home_goals_std'] = home_games['FTHG'].std() if 'FTHG' in df.columns else 1.0

        # Shots
        features['home_shots_for'] = home_games['HS'].mean() if 'HS' in df.columns else 12.0
        features['home_shots_against'] = home_games['AS'].mean() if 'AS' in df.columns else 10.0
        features['home_sot_for'] = home_games['HST'].mean() if 'HST' in df.columns else 4.0
        features['home_sot_against'] = home_games['AST'].mean() if 'AST' in df.columns else 3.0

        # Shot accuracy
        shots = home_games['HS'].sum() if 'HS' in df.columns else 1
        features['home_shot_accuracy'] = (home_games['HST'].sum() / shots) if shots > 0 else 0.33

        # Corners
        features['home_corners_for'] = home_games['HC'].mean() if 'HC' in df.columns else 5.0
        
        # Goals volatility & consistency
        features['home_goals_cv'] = (home_games['FTHG'].std() / (home_games['FTHG'].mean() + 0.001)) if 'FTHG' in df.columns else 0.5
        features['home_consistency'] = 1 - features['home_goals_cv']
        
        # Goal balance
        features['home_goal_balance'] = features['home_goals_for'] - features['home_goals_against']
        
        # Conversion rate
        features['home_conversion_rate'] = features['home_goals_for'] / (features['home_sot_for'] + 0.001)
    else:
        # Default values
        features['home_goals_for'] = 1.5
        features['home_goals_against'] = 1.2
        features['home_goals_std'] = 1.0
        features['home_shots_for'] = 12.0
        features['home_shots_against'] = 10.0
        features['home_sot_for'] = 4.0
        features['home_sot_against'] = 3.0
        features['home_shot_accuracy'] = 0.33
        features['home_corners_for'] = 5.0
        features['home_goals_cv'] = 0.5
        features['home_consistency'] = 0.5
        features['home_goal_balance'] = 0.3
        features['home_conversion_rate'] = 0.375

    # AWAY TEAM FEATURES
    if len(away_games) >= window:
        # Goals
        features['away_goals_for'] = away_games['FTAG'].mean() if 'FTAG' in df.columns else 1.2
        features['away_goals_against'] = away_games['FTHG'].mean() if 'FTHG' in df.columns else 1.5
        features['away_goals_std'] = away_games['FTAG'].std() if 'FTAG' in df.columns else 0.9

        # Shots
        features['away_shots_for'] = away_games['AS'].mean() if 'AS' in df.columns else 10.0
        features['away_shots_against'] = away_games['HS'].mean() if 'HS' in df.columns else 12.0
        features['away_sot_for'] = away_games['AST'].mean() if 'AST' in df.columns else 3.0
        features['away_sot_against'] = away_games['HST'].mean() if 'HST' in df.columns else 4.0

        # Shot accuracy
        shots = away_games['AS'].sum() if 'AS' in df.columns else 1
        features['away_shot_accuracy'] = (away_games['AST'].sum() / shots) if shots > 0 else 0.30

        # Corners
        features['away_corners_for'] = away_games['AC'].mean() if 'AC' in df.columns else 4.0
        
        # Goals volatility & consistency
        features['away_goals_cv'] = (away_games['FTAG'].std() / (away_games['FTAG'].mean() + 0.001)) if 'FTAG' in df.columns else 0.6
        features['away_consistency'] = 1 - features['away_goals_cv']
        
        # Goal balance
        features['away_goal_balance'] = features['away_goals_for'] - features['away_goals_against']
        
        # Conversion rate
        features['away_conversion_rate'] = features['away_goals_for'] / (features['away_sot_for'] + 0.001)
    else:
        # Default values
        features['away_goals_for'] = 1.2
        features['away_goals_against'] = 1.5
        features['away_goals_std'] = 0.9
        features['away_shots_for'] = 10.0
        features['away_shots_against'] = 12.0
        features['away_sot_for'] = 3.0
        features['away_sot_against'] = 4.0
        features['away_shot_accuracy'] = 0.30
        features['away_corners_for'] = 4.0
        features['away_goals_cv'] = 0.6
        features['away_consistency'] = 0.4
        features['away_goal_balance'] = -0.3
        features['away_conversion_rate'] = 0.30

    # COMBINED FEATURES
    features['combined_goals_for'] = features['home_goals_for'] + features['away_goals_for']
    features['combined_goals_against'] = features['home_goals_against'] + features['away_goals_against']
    features['combined_shots_for'] = features['home_shots_for'] + features['away_shots_for']
    features['combined_sot_for'] = features['home_sot_for'] + features['away_sot_for']

    # Base derived features
    features['attack_diff'] = features['home_goals_for'] - features['away_goals_against']
    features['defense_diff'] = features['away_goals_for'] - features['home_goals_against']

    # Momentum (3-game vs 7-game average)
    if len(home_games_3) >= 3:
        home_goals_for_3 = home_games_3['FTHG'].mean() if 'FTHG' in df.columns else features['home_goals_for']
    else:
        home_goals_for_3 = features['home_goals_for']
    
    if len(away_games_3) >= 3:
        away_goals_for_3 = away_games_3['FTAG'].mean() if 'FTAG' in df.columns else features['away_goals_for']
    else:
        away_goals_for_3 = features['away_goals_for']
    
    features['home_momentum'] = home_goals_for_3 - features['home_goals_for']
    features['away_momentum'] = away_goals_for_3 - features['away_goals_for']
    features['combined_momentum'] = features['home_momentum'] + features['away_momentum']

    # =========================================================================
    # EXTERNAL FEATURES
    # =========================================================================
    
    # 1. WIN/LOSS STREAKS (wins and losses in last N games)
    if len(home_games) >= window:
        home_w = ((home_games['FTHG'] > home_games['FTAG']).sum())
        home_l = ((home_games['FTHG'] < home_games['FTAG']).sum())
    else:
        home_w = window // 2
        home_l = window // 2
    
    if len(away_games) >= window:
        away_w = ((away_games['FTAG'] > away_games['FTHG']).sum())
        away_l = ((away_games['FTAG'] < away_games['FTHG']).sum())
    else:
        away_w = window // 2
        away_l = window // 2
    
    features['home_wins'] = home_w
    features['home_losses'] = home_l
    features['away_wins'] = away_w
    features['away_losses'] = away_l
    features['home_win_rate'] = home_w / (home_w + home_l + 0.001)
    features['away_win_rate'] = away_w / (away_w + away_l + 0.001)

    # 2. RECENT GOAL-SCORING STREAKS (high-scoring games in last 5 matches)
    high_scoring_threshold = 2.5
    if len(home_games) >= 5:
        home_high = ((home_games.tail(5)['FTHG'] + home_games.tail(5)['FTAG']) > high_scoring_threshold).sum() / 5
    else:
        home_high = 0.5
    
    if len(away_games) >= 5:
        away_high = ((away_games.tail(5)['FTAG'] + away_games.tail(5)['FTHG']) > high_scoring_threshold).sum() / 5
    else:
        away_high = 0.5
    
    features['home_high_scoring_streak'] = home_high
    features['away_high_scoring_streak'] = away_high

    # 3. HEAD-TO-HEAD STATISTICS (O/U record between teams)
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ]
    
    if len(h2h) >= 3:
        features['h2h_over_pct'] = ((h2h['FTHG'] + h2h['FTAG']) > 2.5).mean()
    else:
        features['h2h_over_pct'] = 0.5  # Default to 50%

    # 4. LEAGUE-WIDE TREND (rolling mean of goals - last 10 matches in data)
    if len(df) > 0:
        league_recent = df.tail(10)
        features['league_goal_trend'] = (league_recent['FTHG'] + league_recent['FTAG']).mean()
    else:
        features['league_goal_trend'] = 2.6

    # 5. DEFENSIVE METRICS
    if len(home_games) >= window:
        features['home_shots_conceded_avg'] = home_games['AS'].mean() if 'AS' in df.columns else 10.0
        features['home_defensive_variance'] = home_games['FTAG'].std() if 'FTAG' in df.columns else 1.0
    else:
        features['home_shots_conceded_avg'] = 10.0
        features['home_defensive_variance'] = 1.0
    
    if len(away_games) >= window:
        features['away_shots_conceded_avg'] = away_games['HS'].mean() if 'HS' in df.columns else 12.0
        features['away_defensive_variance'] = away_games['FTHG'].std() if 'FTHG' in df.columns else 1.1
    else:
        features['away_shots_conceded_avg'] = 12.0
        features['away_defensive_variance'] = 1.1

    return features