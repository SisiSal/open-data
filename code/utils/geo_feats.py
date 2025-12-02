import math
import numpy as np

#Second distance by default is opponent's goal
def calculate_distance_coordinates(x1, y1, x2=120.0, y2=40.0):
    diff_sqr_x = (x2 - x1)**2
    diff_sqr_y = (y2 - y1)**2
    distance = math.sqrt(diff_sqr_x + diff_sqr_y)

    return distance

def calculate_post_angle(x, y, g1_x=120.0, g1_y=36.0, g2_x=120.0, g2_y=44.0):
    if x == 120.0 and (36.0 <= y <= 44.0):
        return 180

    if x == 120.0 and (y > 36.0 or y < 44.0):
        return 0

    ## calculating the three sides of the triangle.
    up_dis = calculate_distance_coordinates(x, y, g1_x, g1_y)
    down_dis = calculate_distance_coordinates(x, y, g2_x, g2_y)
    posts_dis = calculate_distance_coordinates(g1_x, g1_y, g2_x, g2_y)

    ## using cosine law
    value = ((up_dis**2) + (down_dis**2) - (posts_dis**2)) / (2 * up_dis * down_dis)

    angle = np.degrees(np.arccos(value))

    return angle

def area(x1, y1, x2, y2, x3, y3): 
    """
    Funtion to calculate area of triangle.

    Args:
        float: coordinates for triangle vertices.

    Returns:
        float: area of the triangle.
    """    
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)  
                + x3 * (y1 - y2)) / 2.0) 

def is_inside(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_1_x=120.0, pole_1_y=36.0, pole_2_x=120.0, pole_2_y=44.0):
    """
    Function to return whether player is between the player taking shot and goal.

    Args:
        player_coord_x (float): player-coordinate-x.
        player_coord_y (float): player-coordinate-y.
        shot_location_x (float): shot-coordinate-x.
        shot_location_y (float): shot-coordinate-y.
        pole_1_x (float, optional): goal-post(1) coordinate x. Defaults to 120.
        pole_1_y (float, optional): goal-post(1) coordinate y. Defaults to 36.
        pole_2_x (float, optional): goal-post(2) coordinate x. Defaults to 120.0.
        pole_2_y (float, optional): goal-post(2) coordinate x. Defaults to 44.
    
    Returns:
        bool: True if present else False.
    """    
    # calculate area of triangle ABC 
    A = area(shot_location_x, shot_location_y, pole_1_x, pole_1_y, pole_2_x, pole_2_y) 
  
    # calculate area of triangle PBC  
    A1 = area(player_coord_x, player_coord_y, pole_1_x, pole_1_y, pole_2_x, pole_2_y) 
      
    # calculate area of triangle PAC  
    A2 = area(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_2_x, pole_2_y) 
      
    # calculate area of triangle PAB  
    A3 = area(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_1_x, pole_1_y) 
      
    # check if sum of A1, A2 and A3  
    # is same as A 
    if round(A,2) == round(A1 + A2 + A3, 2): 
        return True
    else: 
        return False
    

def freeze_frame_vars(freeze_frame, shot_location_x, shot_location_y):
    """
    Function for making freeze frame variables.

    Args:
        freeze_frame (list): containing tracking information.
        shot_location_x (float): shot coordinate location x.
        shot_location_y (float): shot coordinate location y.

    Returns:
        float values: 1. number of teammates between goal and shot-location.
                      2. number of opponents(excluding goalkeeper) between goal and shot-location.
                      3. goalkeeper covering angle.
                      4. distance between goalkeeper and the goal.
                      5. distance between goalkeeper and the shot-location.
    """    
    ## init two variable to 0
    count_teammate, count_opponent, goal_keeper_angle, dis_goal_keeper, dis_shot_keeper = 0, 0, 0, 0, 0

    ## traverse the freeze frame
    for frame in freeze_frame:
        ## fetch coodinate location of the players
        x_coord = frame["location"][0]
        y_coord = frame["location"][1]

        ## fetch player's position
        position = frame["position"]["name"]

        if position != "Goalkeeper":
            if frame["teammate"] == True and is_inside(x_coord, y_coord, shot_location_x, shot_location_y):
                count_teammate += 1
            
            elif frame["teammate"] == False and is_inside(x_coord, y_coord, shot_location_x, shot_location_y):
                count_opponent += 1
        else:
            ## compute goalkeeper covering angle
            goal_keeper_angle = calculate_post_angle(x_coord, y_coord)

            ## compute distance between goalkeeper and goal
            dis_goal_keeper = calculate_distance_coordinates(x_coord, y_coord)

            ## compute distance between goalkeeper and shot-location
            dis_shot_keeper = calculate_distance_coordinates(x_coord, y_coord, shot_location_x, shot_location_y)
    
    return count_teammate, count_opponent, goal_keeper_angle, dis_goal_keeper, dis_shot_keeper

def add_freeze_frame_vars(df):
    """
    Function to add freeze frame variables to dataframe.

    Args:
        df (DataFrame): input dataframe.

    Returns:
        DataFrame: dataframe with added freeze frame variables.
    """    
        # Initialize lists to store new variables
    teammates_between = []
    opponents_between = []
    gk_angle = []
    gk_dist_goal = []
    gk_dist_shot = []

    # Loop through each row to compute freeze-frame variables
    for i, row in df.iterrows():
        freeze_frame = row["shot_freeze_frame"]
        shot_x = row["loc_x"]
        shot_y = row["loc_y"]

        # Call your existing function
        count_teammate, count_opponent, goal_keeper_angle, dis_goal_keeper, dis_shot_keeper = freeze_frame_vars(
            freeze_frame, shot_x, shot_y
        )

        teammates_between.append(count_teammate)
        opponents_between.append(count_opponent)
        gk_angle.append(goal_keeper_angle)
        gk_dist_goal.append(dis_goal_keeper)
        gk_dist_shot.append(dis_shot_keeper)
    
    df["num_teammate_in_between"] = teammates_between
    df["num_opponent_in_between"] = opponents_between
    df['player_in_between'] = df["num_opponent_in_between"] + df["num_teammate_in_between"]
    df["goal_keeper_angle"] = gk_angle
    df["dist_goal_keeper"] = gk_dist_goal
    df["dist_shot_keeper"] = gk_dist_shot
    return df