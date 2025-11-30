import math
import numpy as np

def calculate_post_angle(x, y, g1_x=104, g1_y=30.34, g2_x=104, g2_y=37.66):
    if x == 104 and (30.34 <= y <= 37.66):
        return 180

    if x == 104 and (y > 37.66 or y < 30.34):
        return 0

    ## calculating the three sides of the triangle.
    up_dis = calculate_distance_coordinates(x, y, g1_x, g1_y)
    down_dis = calculate_distance_coordinates(x, y, g2_x, g2_y)
    posts_dis = calculate_distance_coordinates(g1_x, g1_y, g2_x, g2_y)

    ## using cosine law
    value = ((up_dis**2) + (down_dis**2) - (posts_dis**2)) / (2 * up_dis * down_dis)

    angle = np.degrees(np.arccos(value))

    return angle

#Second distance by default is opponent's goal
def calculate_distance_coordinates(x1, y1, x2=104.0, y2=34.0):
    diff_sqr_x = (x2 - x1)**2
    diff_sqr_y = (y2 - y1)**2
    distance = math.sqrt(diff_sqr_x + diff_sqr_y)

    return distance

def change_dims(old_value, old_min, old_max, new_min, new_max):
    '''
    Function for changing the coordinates to our pitch dimensions.

    Arguments:
        old_value, old_min, old_max, new_min, new_max -- float values.

    Returns:
        new_value -- float value(the coordinate value either x or y).
    '''
    ## calculate the value
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

    return new_value

def coordinates_x(value):
    '''
    Return x coordinate
    '''
    value_x = change_dims(value[0], 0, 120, 0, 104)
    return value_x

def coordinates_y(value):
    '''
    Return 80 - x coordinate
    '''
    value_y = change_dims(80- value[1], 0, 80, 0, 68)
    return value_y

#area of triangle from shot and the two goal posts
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

def is_inside(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_1_x=104.0, pole_1_y=30.34, pole_2_x=104.0, pole_2_y=37.66):
    """
    Function to return whether player is between the player taking shot and goal.

    Args:
        player_coord_x (float): player-coordinate-x.
        player_coord_y (float): player-coordinate-y.
        shot_location_x (float): shot-coordinate-x.
        shot_location_y (float): shot-coordinate-y.
        pole_1_x (float, optional): goal-post(1) coordinate x. Defaults to 104.0.
        pole_1_y (float, optional): goal-post(1) coordinate y. Defaults to 30.34.
        pole_2_x (float, optional): goal-post(2) coordinate x. Defaults to 104.0.
        pole_2_y (float, optional): goal-post(2) coordinate x. Defaults to 37.66.
    
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
        x_coord = coordinates_x(frame["location"])
        y_coord = coordinates_y(frame["location"])

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