import ast
import pandas as pd
import numpy as np
from Code.utils.contextual_feats import calculate_score_per_match, calculate_player_on_pitch
from Code.utils.geo_feats import calculate_distance_coordinates, calculate_post_angle, freeze_frame_vars
from Code.utils.data_cleaning import add_pass_type, filter_rows, remove_empty_cols, remove_invalid_time, drop_predef_cols, fill_predef_cols

print('Starting filtering files...')
df = filter_rows()
print('Finished.')

print('Calculating time on pitch...')
df = calculate_player_on_pitch(df)
print('Finished.')

print('Cleaning rows and cols...')
df = remove_invalid_time(df)
df = drop_predef_cols(df)
df = fill_predef_cols(df)
print('Finished.')

df_test = df.copy()
## add x and y coordinate columns
df_test['loc_x'] = df_test['location'].apply(lambda x: x[0])
df_test['loc_y'] = df_test['location'].apply(lambda x: x[1])

## create distance and angle columns
df_test['dist_to_post'] = df_test.apply(lambda x: calculate_distance_coordinates(x['loc_x'], x['loc_y']), axis=1)
df_test['angle'] = df_test.apply(lambda x: calculate_post_angle(x['loc_x'], x['loc_y']), axis=1)


print('Extracting position X and Y from location column...')
df['loc_x'] = df['location'].apply(lambda x: x[0])
df['loc_y'] = df['location'].apply(lambda x: x[1])
print('Finished.')

print('Calculating distance to post...')
vector_cal_dis_cordinates = np.vectorize(calculate_distance_coordinates)
df['dist_to_post'] = vector_cal_dis_cordinates(df['loc_x'], df['loc_y'])
print('Finished.')

print('Calculating angle to post...')
vector_cal_post_angle = np.vectorize(calculate_post_angle)
df['angle_to_post'] = vector_cal_post_angle(df['loc_x'], df['loc_y'])
print('Finished.')

print('Calculating partial scores...')
df = df.sort_values(['match_id', 'period', 'index'])
df = df.groupby('match_id', group_keys = False).apply(calculate_score_per_match)
print('Finished.')

#df = remove_empty_cols(df)

df.to_csv('processed_events.csv', index=False)


def compute_player_in_between(row):
    freeze = row["shot_freeze_frame"]
    x = row["location"].apply(lambda x: x[0])
    y = row["location"].apply(lambda x: x[1])

    count_teammate, count_opponent, _, _, _ = freeze_frame_vars(freeze, x, y)
    return count_teammate + count_opponent


def compute_goalkeeper_angle(row):
    freeze = row["shot_freeze_frame"]
    x = row["location"].apply(lambda x: x[0])
    y = row["location"].apply(lambda x: x[1])

    _, _, gk_angle, _, _ = freeze_frame_vars(freeze, x, y)
    return gk_angle

print("Adding player_in_between...")
df["player_in_between"] = df.apply(compute_player_in_between, axis=1)
print('Finished.')

print("Adding goal_keeper_angle...")
df["goal_keeper_angle"] = df.apply(compute_goalkeeper_angle, axis=1)
print('Finished.')
