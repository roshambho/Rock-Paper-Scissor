import math

def list_of_id():
    """
    Generate a list of all unique combinations of indices from 0 to 20 (inclusive).
    """
    ids = [i for i in range(21)]
    id_combinations = [(i, i+j+1) for i in ids for j in range(len(ids)-(i+1))]
    return id_combinations

def distance_2d(p1, p2):
    """
    Calculate the Euclidean distance between two 2D points.
    """
    distance = math.sqrt(
        math.pow(p1[0]-p2[0], 2) +
        math.pow(p1[1]-p2[1], 2)
    )
    return int(distance)

def calc_all_distance(height, width, landmark_list):
    """
    Calculate distances between all pairs of hand landmarks.
    """
    if landmark_list:
        my_list = list_of_id()
        all_distance = []
        for item in my_list:
            point_1 = [landmark_list[item[0]][0]*height, landmark_list[item[0]][1]*width]
            point_2 = [landmark_list[item[1]][0]*height, landmark_list[item[1]][1]*width]
            distance = distance_2d(point_1, point_2)
            all_distance.append(distance)
        return all_distance
    else:
        all_distance = [1] * 210  # If hand not detected, return a fake array to prevent errors
        # This wrong prediction is taken care of since statistical mode of predictions is taken
        return all_distance
