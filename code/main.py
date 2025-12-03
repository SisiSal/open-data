import ast
import pandas as pd
import numpy as np
import Code.utils.contextual_feats as cf
import Code.utils.geo_feats as gf
import Code.utils.data_cleaning as dc
import Code.utils.fig_generation as fg
import Code.utils.split_data as sd
#from Code.utils.contextual_feats import calculate_score_per_match, calculate_player_on_pitch
#from Code.utils.geo_feats import calculate_distance_coordinates, calculate_post_angle, freeze_frame_vars
#from Code.utils.data_cleaning import add_pass_type, filter_rows, remove_empty_cols, remove_invalid_time, drop_predef_cols, fill_predef_cols
import importlib
importlib.reload(gf)
importlib.reload(cf)
importlib.reload(dc)


### Data Preprocessing

print('Starting filtering files...')
df = dc.filter_rows()
print('Finished.')

print('Calculating time on pitch...')
df = cf.calculate_player_on_pitch(df)
df['time_on_field'] = pd.to_timedelta(df['time_on_field'])
print('Finished.')

print('Dropping empty columns...')
df = df.dropna(axis=1, how='all')
print('Finished.')

print('Extracting position X and Y from location column...')
df['loc_x'] = df['location'].apply(lambda x: x[0])
df['loc_y'] = df['location'].apply(lambda x: x[1])
df['loc_x'] = pd.to_numeric(df['loc_x'], errors='coerce')
df['loc_y'] = pd.to_numeric(df['loc_y'], errors='coerce')
print('Finished.')

print('Calculating distance to post...')
vector_cal_dis_cordinates = np.vectorize(gf.calculate_distance_coordinates)
df['dist_to_post'] = vector_cal_dis_cordinates(df['loc_x'], df['loc_y'])
print('Finished.')

print('Calculating angle to post...')
vector_cal_post_angle = np.vectorize(gf.calculate_post_angle)
df['angle_to_post'] = vector_cal_post_angle(df['loc_x'], df['loc_y'])
print('Finished.')

print('Calculating partial scores...')
df = cf.define_home_away_teams(df)
df = df.sort_values(['match_id', 'period', 'index'])
df = df.groupby('match_id', group_keys = False).apply(cf.calculate_score_per_match)
df = cf.add_poss_team_match_state(df)
df = cf.define_game_state_home_or_away(df)
print('Finished.')

print("Adding freeze_frame_vars...")
df = gf.add_freeze_frame_vars(df)
print('Finished.')

print('Adding goal column...')
df['goal'] = np.where(df['shot_outcome_name'] == 'Goal', 1, 0)
print('Finished.')

print('Cleaning rows and cols...')
df = dc.remove_invalid_time(df)
df = dc.drop_predef_cols(df)
df = dc.fill_predef_cols(df)
df['time_on_field'] = df['time_on_field'].dt.total_seconds().fillna(0)
df['time_on_field'] = df['time_on_field'].astype(float)
print('Finished.')

print('hot encoding categorical columns...')
df = dc.hot_encode_categorical_columns(df)
print('Finished.')

print('Creating heatmap...')
fg.generate_heatmap(df)
print('Finished.')

print('Splitting and standardizing...')
X_train, X_test, y_train, y_test = sd.split_data(df)
X_train_scaled, X_test_scaled, scaler = sd.standardize_data(X_train, X_test)
print('Finished.')

df.to_csv('processed_events.csv', index=False)

#df_test = df.copy()
