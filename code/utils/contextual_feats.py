import time
from statsbombpy import sb
import pandas as pd
import numpy as np

def calculate_period_time(period, event_time):
    if period == 1:
        return event_time
    if period == 2:
        return event_time + pd.to_timedelta('00:45:00')
    if period == 3:
        return event_time + pd.to_timedelta('00:90:00')
    if period == 4:
        return event_time + pd.to_timedelta('00:105:00')

def calculate_player_on_pitch(events_df):
    start_time = time.time()

    for index, event in events_df.iterrows():
        #Get event's timestamp, and convert it in hh:mm:ss
        event_time = pd.to_timedelta(event['timestamp'])
        event_time = calculate_period_time(event['period'], event_time)

        #Lineups containt info about player (when they entered the pitch)
        lineup = sb.lineups(event['match_id'])
        team_name = event['team_name']
        lineup = lineup[team_name]

        #Find player from event in lineup, and get its first time in the pitch
        player_init_time = lineup.loc[lineup['player_id'] == event['player_id'], 'positions'].values[0][0]['from']
        player_init_time = pd.to_timedelta('00:' + player_init_time)
        events_df.at[index, 'time_on_field'] = event_time - player_init_time

    end_time = time.time()
    print(f'Total time: {end_time - start_time}')
    events_df.to_csv('checkpoint.csv', index=False)
    return events_df

def calculate_score_per_match(match_df):
    teams = match_df['team_name'].unique()
    team_A = teams[0]
    if len(teams) == 2:
        team_B = teams[1]
    else:
        home_name = match_df.iloc[0]['home_team']
        away_name = match_df.iloc[0]['away_team']
        if team_A == home_name:
            team_B = away_name
        else:
            team_B = home_name
    is_goal = match_df['shot_outcome_name'] == 'Goal'
    goals_team_A = is_goal & (match_df['team_name'] == team_A)
    goals_team_B = is_goal & (match_df['team_name'] == team_B)
    score_A = goals_team_A.cumsum()
    score_B = goals_team_B.cumsum()
    match_df[f'{team_A}'] = score_A.shift(1).fillna(0).astype(int)
    match_df[f'{team_B}'] = score_B.shift(1).fillna(0).astype(int)
    diff = match_df[f'{team_A}'] - match_df[f'{team_B}']

    conditions = [
        diff < 0,
        diff > 0
    ]
    choices = [
        team_B,
        team_A
    ]
    match_df['winning_team'] = np.select(conditions, choices, default = 'draw')
    match_df.drop([f'{team_A}', f'{team_B}'], axis = 1, inplace = True)
    return match_df

def add_poss_team_match_state(df):
    """
    Adds poss_team_match_state efficiently (vectorized).
    """

    df['poss_team_match_state'] = 'opponent'  # default

    df.loc[df['winning_team'] == 'draw', 'poss_team_match_state'] = 'draw'
    df.loc[df['winning_team'] == df['possession_team_name'], 'poss_team_match_state'] = 'possession'

    return df

def define_home_away_teams(events_df):
    competitions = sb.competitions()
    unique_competitions = competitions[['competition_id', 'season_id']].drop_duplicates()
    all_matches = []
    for index, row in unique_competitions.iterrows():
        matches = sb.matches(
            competition_id=row['competition_id'], 
            season_id=row['season_id']
        )
        matches_filtered_columns = matches[['match_id', 'home_team', 'away_team']]
        all_matches.append(matches_filtered_columns)
    matches_df = pd.concat(all_matches, ignore_index=True)
    matches_df['match_id'] = matches_df['match_id'].astype(int)
    events_df['match_id'] = events_df['match_id'].astype(int)
    events_df = events_df.merge(matches_df, on='match_id', how='left')
    print(events_df.columns)
    events_df['venue'] = np.where(
        events_df['team_name'] == events_df['home_team'],
        'home',
        'away'
    )
    return events_df

def define_game_state_home_or_away(events_df):
    conditions = [
        (events_df['winning_team'] == 'draw'),
        (events_df['winning_team'] == events_df['home_team']),
        (events_df['winning_team'] == events_df['away_team'])
    ]
    choices = ['Draw', 'Home', 'Away']
    events_df['game_state'] = np.select(conditions, choices, default='Unknown')
    return events_df