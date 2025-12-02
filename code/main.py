import ast
import pandas as pd
import numpy as np
import utils.contextual_feats as cf
import utils.geo_feats as gf
import utils.data_cleaning as dc
#from Code.utils.contextual_feats import calculate_score_per_match, calculate_player_on_pitch
#from Code.utils.geo_feats import calculate_distance_coordinates, calculate_post_angle, freeze_frame_vars
#from Code.utils.data_cleaning import add_pass_type, filter_rows, remove_empty_cols, remove_invalid_time, drop_predef_cols, fill_predef_cols
import importlib
importlib.reload(gf)
importlib.reload(cf)
importlib.reload(dc)

print('Starting filtering files...')
df = dc.filter_rows()
print('Finished.')

print('Calculating time on pitch...')
#df = cf.calculate_player_on_pitch(df)
print('Finished.')

#drop all columns with only empty values to replace this next function
#df = dc.remove_empty_cols(df)
print('Dropping empty columns...')
df = df.dropna(axis=1, how='all')
print('Finished.')

print('Extracting position X and Y from location column...')
df['loc_x'] = df['location'].apply(lambda x: x[0])
df['loc_y'] = df['location'].apply(lambda x: x[1])
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
print('Finished.')

print('Cleaning rows and cols...')
#df = dc.remove_invalid_time(df)
df = dc.drop_predef_cols(df)
df = dc.fill_predef_cols(df)
print('Finished.')


print("Adding freeze_frame_vars...")
df = gf.add_freeze_frame_vars(df)
print('Finished.')

df.to_csv('processed_events.csv', index=False)


#df_test = df.copy()