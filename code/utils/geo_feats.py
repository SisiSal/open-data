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