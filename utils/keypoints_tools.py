from scipy.spatial.distance import euclidean
import copy
from sklearn.preprocessing import StandardScaler
import math

def remove_Points(df, points):
    df['keypoints_clear'] = df['keypoints'].apply(lambda x: [point for i, point in enumerate(x) if i not in points])
    df['keypoint_scores_clear'] = df['keypoint_scores'].apply(lambda x: [score for i, score in enumerate(x) if i not in points])
    return df


def count_lenghts(keypoints, connections):
    lenghts = [euclidean(keypoints[point1], keypoints[point2]) for point1, point2 in connections]
    return lenghts

def add_lenghts(df, connections, useless_points=[]):
    connections = [(x, y) for x, y in connections if x not in useless_points and y not in useless_points]
    df['lengths'] = df['keypoints'].apply(lambda x: count_lenghts(x, connections))
    
def add_relative_lenghts(df, base_connestion):
    df['base_length'] = df['keypoints'].apply(lambda x: count_lenghts(x, [base_connestion])[0])
    # gdyby oba stawy w tym samym pixelu (w 1percent 236 razy)
    df['base_length'] = df['base_length'].apply(lambda x: 1 if x < 1 else x)
    df['relative_lengths'] = df.apply(lambda row: [length / row['base_length'] for length in row['lengths']], axis=1)
    
def change_value(list_to_change, index, value):
    list_to_change[index]= value

    return list_to_change

def std_index(kolumna, index): 
    elements = kolumna.apply(lambda x: x[index])
    elements_mean = elements.mean()
    elements_std = elements.std()
    elements = [(elem - elements_mean) / elements_std for elem in elements]
    return kolumna.apply(lambda x: change_value(x, index, elements[index]))
 
def std_column(df, column_name, new_column_name):
# Standaryzacja dla wszystkich indeksów
    lenghts_list_len = len(df[column_name][0])
    # Załóż, że wszystkie listy mają tę samą długość
    df[new_column_name] = df[column_name].apply(copy.deepcopy)
    for i in range(lenghts_list_len):
        
        df[new_column_name] = std_index(df[new_column_name], i)
        
def angle3pt_3d(a, b, c):
    # Oblicz różnice współrzędnych dla każdej osi
    delta_x1, delta_y1, delta_z1 = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    delta_x2, delta_y2, delta_z2 = c[0] - b[0], c[1] - b[1], c[2] - b[2]

    # Oblicz kąty azymutu (phi) i elewacji (theta) dla obu wektorów
    theta1 = math.atan2(delta_y1, math.sqrt(delta_x1**2 + delta_z1**2))
    phi1 = math.atan2(delta_z1, delta_x1)
    
    theta2 = math.atan2(delta_y2, math.sqrt(delta_x2**2 + delta_z2**2))
    phi2 = math.atan2(delta_z2, delta_x2)

    # Oblicz kąt między wektorami (różnice kątów azymutu i elewacji)
    angle_rad = math.acos(math.cos(theta1 - theta2) * math.cos(phi1 - phi2))

    # Konwertuj kąt z radianów na stopnie
    ang = math.degrees(angle_rad)
    return ang + 360 if ang < 0 else ang

def angle3pt_2d(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def get_base_angle(mid_point, end_point, three_dim=False):
    base_end_point = [0.0, mid_point[1], mid_point[2]] if three_dim else [0.0, mid_point[1]]
    result = angle3pt_3d(base_end_point, mid_point, end_point) if three_dim else angle3pt_2d(base_end_point, mid_point, end_point)
    return result

# dla base_points nie działa tak jak powinno
def get_angle3(a, b, c, base_points=None, three_dim=False):
    if base_points:
        base_angle = get_base_angle(base_points[0], base_points[1], three_dim)
    else:
        base_angle = 0.0
        
    if three_dim:
        angle = angle3pt_3d(a, b, c) - base_angle
    else:
        angle = angle3pt_2d(a, b, c) - base_angle
    return angle + 360 if angle < 0 else angle

def count_angles(keypoints, angles, base_points=None, three_dim=False):
    angles_size = [get_angle3(keypoints[point1], keypoints[point2], keypoints[point3], base_points, three_dim) for point1, point2, point3 in angles]
    return angles_size

def add_angles(df, angles, useless_points=[], base_connestion=None, three_dim=False):
    angles = [(x, y, z) for x, y, z in angles if x not in useless_points and y not in useless_points]
    df['angles'] = df['keypoints'].apply(lambda x: count_angles(x, angles, base_connestion, three_dim))
    
def count_variant_angle(keypoints, connections, three_dim=False):
    lenghts = [get_base_angle(keypoints[point1], keypoints[point2], three_dim=three_dim) for point1, point2 in connections]
    return lenghts

def add_variant_angle(df, connections, useless_points=[], three_dim=False):
    connections = [(x, y) for x, y in connections if x not in useless_points and y not in useless_points]
    df['variant_angles'] = df['keypoints'].apply(lambda x: count_variant_angle(x, connections, three_dim))