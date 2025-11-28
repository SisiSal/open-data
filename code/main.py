import ast
import pandas as pd
import numpy as np
from utils.contextual_feats import calculate_score_per_match, calculate_player_on_pitch
from utils.geo_feats import calculate_distance_coordinates, calculate_post_angle
from utils.data_cleaning import filter_rows, remove_empty_cols, remove_invalid_time, remove_not_open_play, drop_predef_cols, fill_predef_cols

print('Starting filtering files...')
df = filter_rows()
df = remove_not_open_play(df)
print('Finished.')

print('Calculating time on pitch...')
df = calculate_player_on_pitch(df)
print('Finished.')

print('Cleaning rows and cols...')
df = remove_invalid_time(df)
df = drop_predef_cols(df)
df = fill_predef_cols(df)
print('Finished.')

print('Extracting position X and Y from location column...')
def safe_str_to_list(value):
    """
    Converts string representations of lists/tuples into actual Python lists.
    Returns [NaN, NaN] if the parsing fails.
    """
    if isinstance(value, str):
        try:
            result = ast.literal_eval(value)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                return list(result)
        except (ValueError, SyntaxError):
            pass 
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return list(value)    
    return [np.nan, np.nan] # Fallback for bad/missing data

df['location'] = df['location'].apply(safe_str_to_list)
coordinates = np.array(df['location'].to_list())
df['loc_x'] = coordinates[:, 0]
df['loc_y'] = coordinates[:, 1]
df.drop(['location'], axis = 1, inplace = True)
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
