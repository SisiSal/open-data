import os
import json
import pandas as pd

path = "data/events"
all_events = []

def filter_rows():
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.json_normalize(data, sep='_')
            df_shots = df[
                (df['type_name'] == 'Shot') &
                (df['type_name'].notna())].copy()
            if not df_shots.empty:
                df_shots['match_id'] = filename.replace('.json', '')
                all_events.append(df_shots)

    events_df = pd.concat(all_events, ignore_index=True)
    print(events_df.shape)
    print(events_df.head())
    print(events_df["shot_type_name"].value_counts())
    return events_df

def remove_invalid_time(df):
    df['time_on_field'] = pd.to_timedelta(df['time_on_field'], errors='coerce')
    time_on_field_invalid = df[df['time_on_field'] <= pd.Timedelta(0)].index
    df.drop(time_on_field_invalid, inplace=True)
    return df

def remove_not_open_play(df):
    not_open_plays = df[df['shot_type_name'] != 'Open Play'].index
    df.drop(not_open_plays, inplace=True)
    return df

def remove_empty_cols(df):
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 0].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def drop_predef_cols(df):
    df.drop(['player_id', 'player_name', 'position_id', 'position_name', 'off_camera', 'shot_end_location'], axis = 1, inplace = True)
    df.drop(['shot_technique_name', 'shot_type_id', 'shot_type_name'], axis = 1, inplace = True)
    df.drop(['out', 'shot_saved_to_post', 'shot_deflected', 'shot_saved_off_target'], axis = 1, inplace = True)
    return df

def fill_predef_cols(df):
    fill_false = ['shot_open_goal', 'shot_follows_dribble', 'shot_redirect', 'under_pressure', 'shot_aerial_won', 'shot_one_on_one']
    for col in fill_false:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df
