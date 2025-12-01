import os
import json
import pandas as pd


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
            df_shots = df[
                (df['shot_type_name'] == 'Open Play') &
                (df['shot_type_name'].notna())].copy()
            if not df_shots.empty:
                df_shots['match_id'] = filename.replace('.json', '')
                all_events.append(df_shots)

    events_df = pd.concat(all_events, ignore_index=True)
    print(events_df.shape)
    print(events_df.head())
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
    df.drop(['player_id', 'player_name', 'position_id', 'position_name', 'off_camera', 'shot_end_location'], axis = 1, inplace = True)
    df.drop(['shot_technique_name', 'shot_type_id', 'shot_type_name'], axis = 1, inplace = True)
    df.drop(['out', 'shot_saved_to_post', 'shot_saved_off_target'], axis = 1, inplace = True)
    return df

def fill_predef_cols(df):
    fill_false = ['shot_open_goal', 'shot_follows_dribble', 'shot_redirect', 'under_pressure', 'shot_aerial_won', 'shot_one_on_one']
    for col in fill_false:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df
