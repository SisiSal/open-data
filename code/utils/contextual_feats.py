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
        team_B = 'other_team'
        print(f'match {match_df.iloc[0]['match_id']}')
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


