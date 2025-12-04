import ast
import pandas as pd
import numpy as np
import utils.contextual_feats as cf
import utils.geo_feats as gf
import utils.data_cleaning as dc
import utils.fig_generation as fg
import utils.split_data as sd
import utils.grid_search as gs
import utils.models as mod
import importlib
importlib.reload(gf)
importlib.reload(cf)
importlib.reload(dc)
importlib.reload(fg)
importlib.reload(sd)
importlib.reload(gs)
importlib.reload(mod)


### Data Preprocessing


#print('statsbomb evaluation...')
#eval_df = pd.read_csv('processed_events.csv')
#eval_X_train, eval_X_test, eval_y_train, eval_y_test = sd.split_data(eval_df)
#mod.evaluate_statsbomb(eval_X_test)

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


df.to_csv('processed_events.csv', index=False)


print('Creating heatmap...')
fg.generate_heatmap(df)
print('Finished.')

print('Splitting and standardizing...')
X_train, X_test, y_train, y_test = sd.split_data(df)
X_train_scaled, X_test_scaled, scaler = sd.standardize_data(X_train, X_test)
print('Finished.')

print(df.columns)
print('Calculating VIF...')
X_numeric = df.drop(columns=['goal','match_id','shot_statsbomb_xg', 'id'], axis=1)
X_numeric = df[['duration_buildup_shot', 'distance_buildup_shot', 'time_on_field', 
                'dist_to_post', 'angle_to_post', 'player_in_between',
                'goal_keeper_angle', 'dist_goal_keeper', 'dist_shot_keeper']]
vif_table = sd.calculate_vif(X_numeric)
print(vif_table.sort_values("VIF", ascending=False))
X_numeric.drop(['dist_to_post','goal_keeper_angle'], axis=1,inplace=True)
vif_table = sd.calculate_vif(X_numeric)
print(vif_table.sort_values("VIF", ascending=False))
print('Finished.')

### Modeling Time XD ###

print('Tuning Models...')
best_log_params = gs.tune_log_model(X_train_scaled, y_train)
print('Best Logistic Regression Params:', best_log_params)
#output: Best Logistic Regression Params: {'C': np.float64(10000.0), 'penalty': 'l2', 'solver': 'lbfgs'}
best_rf_params = gs.tune_random_forest(X_train_scaled, y_train)
print('Best Random Forest Params:', best_rf_params)
#output: Best Random Forest Params: {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 7, 'n_estimators': 500}
best_gb_params = gs.tune_xg_boost(X_train_scaled, y_train)
print('Best XGBoost Params:', best_gb_params)
#output: Best XGBoost Params: {'subsample': 1.0, 'n_estimators': 400, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.7}
best_nn_params = gs.tune_neural_network(X_train_scaled, y_train)
print('Best Neural Network Params:', best_nn_params)
#output:
best_nb_params = gs.tune_naive_bayes(X_train_scaled, y_train)
print('Best Naive Bayes Params:', best_nb_params)
#output:
print('Finished.')

print('Add target variable back into scaled data...')
train_df = X_train_scaled.copy()
train_df['goal'] = y_train
test_df = X_test_scaled.copy()
test_df['goal'] = y_test

print('Apply Logistic Regression...')
lr_model, lr_train_df, lr_test_df = mod.log_reg_mod(train_df, test_df, "goal")
print('Apply Random Forest...')
rf_model, rf_train_df, rf_test_df = mod.random_forest_mod(train_df, test_df, "goal")
print('Apply XGBoost...')
gb_model, gb_train_df, gb_test_df = mod.xgboost_mod(train_df, test_df, "goal")


#df_test = df.copy()
