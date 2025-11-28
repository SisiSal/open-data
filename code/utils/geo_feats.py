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
