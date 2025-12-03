import os
import json
import pandas as pd
import numpy as np


def add_pass_type(df):
    '''
    Adds a "pass_type" column to the dataframe based on the type of pass that assisted the shot.    
    Args:
        df (pd.DataFrame): DataFrame containing event data with shot and pass information
    '''
    # create a new column and default value
    df["pass_type"] = "Not Assisted"
    
    # create a dataframe of only passes
    pass_df = df[df["type_name"] == "Pass"].copy().set_index("id")
    
    # shots where shot_key_pass_id is not null
    shot_with_assist = df[df["shot_key_pass_id"].notna()]

    # iterate through each shot with assist
    for idx, row in shot_with_assist.iterrows():
        key_pass_id = row["shot_key_pass_id"]

        # if the key pass doesn't exist, leave pass_type = Not Assisted
        if key_pass_id not in pass_df.index:
            df.at[idx, "pass_type"] = "Other"
            continue
        
        temp_data = pass_df.loc[key_pass_id]
        pass_type = ""

        # Through ball
        if temp_data.get("pass_technique_name") == "Through Ball":
            pass_type = "Through Ball"

        # Cut back
        elif temp_data.get("pass_cut_back") is True:
            pass_type = "Cut Back"

        # Cross
        elif temp_data.get("pass_cross") is True:
            pass_type = "Cross"

        else:
            # Free kicks / corners
            ptype = temp_data.get("pass_type_name")
            if ptype == "Corner":
                pass_type = "From Corner"
            elif ptype == "Free Kick":
                pass_type = "From Free Kick"
            else:
                pass_type = "Other"

        df.at[idx, "pass_type"] = pass_type

def euclid(p1, p2):
    if not isinstance(p1, (list, tuple)) or not isinstance(p2, (list, tuple)):
        return np.nan
    if len(p1) < 2 or len(p2) < 2:
        return np.nan
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def add_duration_buildup_shot(df):
    """
    Adds a column "duration_buildup_shot" to the dataframe, representing the duration (in seconds)
    of the buildup to each shot within the same possession.
    Args:
        df (pd.DataFrame): DataFrame containing event data with shot and possession information
    """

    # compute absolute event time
    df["event_time"] = df["minute"] * 60 + df["second"]

    # initialize output column
    df["duration_buildup_shot"] = np.nan

    # if there are no shots, stop early
    if "Shot" not in df["type_name"].values:
        return

    # group by both possession ID and period so possession doesn't cross periods
    df["possession_key"] = df["possession"].astype(str) + "_" + df["period"].astype(str)

    # precompute possession start times inside the period
    possession_start_times = (
        df.groupby("possession_key")["event_time"].min()
        .to_dict()
    )

    # loop through shot events only
    shot_rows = df[df["type_name"] == "Shot"]

    for idx, row in shot_rows.iterrows():
        key = row["possession_key"]
        start_time = possession_start_times.get(key, None)

        if start_time is not None:
            df.at[idx, "duration_buildup_shot"] = row["event_time"] - start_time

    return

def add_distance_buildup_shot(df):
    """
    Adds a column "distance_buildup_shot" to the dataframe, representing the total distance
    covered during the buildup to each shot within the same possession.
    """

    # event time in seconds
    df["event_time"] = df["minute"] * 60 + df["second"]

    # possession + period composite key
    df["possession_key"] = df["possession"].astype(str) + "_" + df["period"].astype(str)

    df["distance_buildup_shot"] = np.nan

    # process each shot
    shot_rows = df[df["type_name"] == "Shot"]

    for idx, shot in shot_rows.iterrows():

        key = shot["possession_key"]

        # all events in that possession within the same period
        poss_df = df[df["possession_key"] == key].sort_values("event_time")

        # extract xy locations
        locations = poss_df["location"].tolist()

        # compute cumulative distance
        total_dist = 0.0
        for i in range(1, len(locations)):
            total_dist += euclid(locations[i-1], locations[i])

        df.at[idx, "distance_buildup_shot"] = total_dist

path = "data/events"
all_events = []

def filter_rows():
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.json_normalize(data, sep='_')
            add_pass_type(df)
            add_duration_buildup_shot(df)
            add_distance_buildup_shot(df)
            df_shots = df[
                (df['shot_type_name'] == 'Open Play') &
                (df['shot_type_name'].notna())].copy()
            if not df_shots.empty:
                df_shots['match_id'] = filename.replace('.json', '')
                all_events.append(df_shots)

    events_df = pd.concat(all_events, ignore_index=True)
    print(events_df.shape)
    print(events_df.head())
    events_df.to_csv('events_df.csv', index=False)
    return events_df

def remove_invalid_time(df):
    df['time_on_field'] = pd.to_timedelta(df['time_on_field'], errors='coerce')
    time_on_field_invalid = df[df['time_on_field'] <= pd.Timedelta(0)].index
    df.drop(time_on_field_invalid, inplace=True)
    return df

def remove_empty_cols(df):
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 0].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def drop_predef_cols(df):
    cols_to_keep = cols_to_keep = [
        'match_id',
        'goal',
        'under_pressure',
        'shot_first_time',
        'shot_technique_name',
        'shot_body_part_name',
        'shot_aerial_won',
        'pass_type',
        'duration_buildup_shot',
        'distance_buildup_shot',
        'shot_deflected',
        'shot_open_goal',
        'shot_follows_dribble',
        'time_on_field',
        'dist_to_post',
        'angle_to_post',
        'poss_team_match_state',
        'venue',
        'player_in_between',
        'goal_keeper_angle',
        'dist_goal_keeper',
        'dist_shot_keeper'
    ]
    return df[cols_to_keep].copy()

def fill_predef_cols(df):
    fill_false = ['shot_open_goal', 'shot_follows_dribble', 'under_pressure', 'shot_aerial_won', 'shot_first_time', 'shot_deflected']
    df[fill_false] = df[fill_false].fillna(False).astype(int)
    return df

def hot_encode_categorical_columns(df):
    categorical_cols = [
        'shot_technique_name', 
        'shot_body_part_name', 
        'pass_type',
        'poss_team_match_state',
        'venue'
        ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df