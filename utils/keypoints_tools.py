from scipy.spatial.distance import euclidean
import copy
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

def remove_points(df, useless_points):
    df['keypoints'] = df['keypoints'].apply(lambda x: [point for i, point in enumerate(x) if i not in useless_points])
    df['keypoint_scores'] = df['keypoint_scores'].apply(lambda x: [score for i, score in enumerate(x) if i not in useless_points])
    return df


def count_lenghts(keypoints, connections):
    lenghts = [euclidean(keypoints[point1], keypoints[point2]) for point1, point2 in connections]
    return lenghts

def count_lenghts_prob(keypoint_scores, connections):
    lenghts = [keypoint_scores[point1] * keypoint_scores[point2] for point1, point2 in connections]
    return lenghts

def add_lenghts(df, connections, useless_points=[]):
    connections = [(x, y) for x, y in connections if x not in useless_points and y not in useless_points]
    df['lengths'] = df['keypoints'].apply(lambda x: count_lenghts(x, connections))
    df['lengths_prob'] = df['keypoint_scores'].apply(lambda x: count_lenghts_prob(x, connections))
    
def add_relative_lenghts(df, base_connestion):
    df['base_length'] = df['keypoints'].apply(lambda x: count_lenghts(x, [base_connestion])[0])
    # gdyby oba stawy w tym samym pixelu (w 1percent 236 razy)
    df['base_length'] = df['base_length'].apply(lambda x: 1 if x < 1 else x)
    df['relative_lengths'] = df.apply(lambda row: [length / row['base_length'] for length in row['lengths']], axis=1)
    
def change_value(list_to_change, index, value):
    list_to_change[index]= value

    return list_to_change

def std_index(kolumna, index, imput_method): 
    elements = kolumna.apply(lambda x: x[index])
    
    if imput_method == 'mean':
        # Uzupełnij brakujące dane średnią
        elements_mean = elements.mean()
        elements = elements.fillna(elements_mean)
    if imput_method == 'median':
        # Uzupełnij brakujące dane średnią
        elements_median = elements.median()
        elements = elements.fillna(elements_median)
    if imput_method == 'mode':
        # Uzupełnij brakujące dane średnią
        elements_median = elements.mode().iloc[0]
        elements = elements.fillna(elements_median)
    elif imput_method == 'most':
        # imput beetwen Q1 and Q3
        mean = elements.mean()
        std = elements.std()
        is_null = elements.isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_values = np.random.randint(mean - std, mean + std, size = is_null)
        # fill NaN values in Age column with random values generated
        values_slice = elements.copy()
        values_slice[np.isnan(values_slice)] = rand_values
        elements = values_slice
        
    elements_mean = elements.mean()
    elements_std = elements.std()
    elements = [(elem - elements_mean) / elements_std for elem in elements]
    elements = iter(elements)
    return kolumna.apply(lambda x: change_value(x, index, next(elements)))

    
def drop_low_prob(row, column_name, column_prob, prob_drop_value):
    return [value if prob >= prob_drop_value else np.nan for value, prob in zip(row[column_name], row[column_prob])]

def exclude_low_prob(df, column_name, column_prob, prob_drop_value):
    # Sprawdź, czy nazwy kolumn istnieją w ramce danych
    df[column_name + f'_{prob_drop_value}'] = df.apply(drop_low_prob, axis=1, args=(column_name, column_prob, prob_drop_value))

def std_column(df, column_name, new_column_name, imput_method='mean'):
    # Standaryzacja dla wszystkich indeksów
    # Załóż, że wszystkie listy mają tę samą długość
    lenghts_list_len = len(df[column_name][0])
    
    df[new_column_name] = df[column_name].apply(copy.deepcopy)

    for i in range(lenghts_list_len):
        
        df[new_column_name] = std_index(df[new_column_name], i, imput_method)
        
def bin_index(kolumna, index, discretizer): 
    elements = kolumna.apply(lambda x: x[index]).to_numpy().reshape(-1, 1)
    
    elements = discretizer.fit_transform(elements).reshape(-1)
    elements = iter(elements)
    return kolumna.apply(lambda x: change_value(x, index, next(elements)))
       
def bin_column(df, column_name, new_column_name, discretizerType, discretizer_args):
    # Standaryzacja dla wszystkich indeksów
    # Załóż, że wszystkie listy mają tę samą długość
    lenghts_list_len = len(df[column_name][0])
    
    df[new_column_name] = df[column_name].apply(copy.deepcopy)
    discretizer = discretizerType(**discretizer_args)
    binarized_result = discretizer.fit_transform(np.array([np.array(row) for row in df[new_column_name]]))
    df[new_column_name] = [list(row) for row in binarized_result]
    
        
        
"""def bin_column(df, column_name, new_column_name, discretizerType, discretizer_args):
    # Standaryzacja dla wszystkich indeksów
    # Załóż, że wszystkie listy mają tę samą długość
    lenghts_list_len = len(df[column_name][0])
    
    df[new_column_name] = df[column_name].apply(copy.deepcopy)

    for i in range(lenghts_list_len):
        discretizer = discretizerType(**discretizer_args)
        df[new_column_name] = bin_index(df[new_column_name], i, discretizer)"""
        
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
    angle = (angle + 360) if angle < 0 else angle
    #angle = angle if angle <= 180 else abs(angle - 360)
    return angle 

def angle3pt_2d(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    angle = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    angle = (angle + 360) if angle < 0 else angle
    #angle = angle if angle <= 180 else abs(angle - 360)
    return angle

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
    angle = (angle + 360) if angle < 0 else angle
    #angle = angle if angle <= 180 else abs(angle - 360)
    return angle

def count_angles(keypoints, angles, base_points=None, three_dim=False):
    angles_size = [get_angle3(keypoints[point1], keypoints[point2], keypoints[point3], base_points, three_dim) for point1, point2, point3 in angles]
    return angles_size

def count_angles_prob(keypoint_scores, angles):
    angles_size = [keypoint_scores[point1] * keypoint_scores[point2] * keypoint_scores[point3] for point1, point2, point3 in angles]
    return angles_size

def add_angles(df, angles, useless_points=[], base_connestion=None, three_dim=False):
    angles = [(x, y, z) for x, y, z in angles if x not in useless_points and y not in useless_points]
    df['angles'] = df['keypoints'].apply(lambda x: count_angles(x, angles, base_connestion, three_dim))
    df['angles_prob'] = df['keypoint_scores'].apply(lambda x: count_angles_prob(x, angles))
    
def count_variant_angle(keypoints, connections, three_dim=False):
    lenghts = [get_base_angle(keypoints[point1], keypoints[point2], three_dim=three_dim) for point1, point2 in connections]
    return lenghts

def add_variant_angle(df, connections, useless_points=[], three_dim=False):
    connections = [(x, y) for x, y in connections if x not in useless_points and y not in useless_points]
    df['variant_angles'] = df['keypoints'].apply(lambda x: count_variant_angle(x, connections, three_dim))
    # should refactor name, it's just for 2 points, and count_angles_prob is for 3 points. here 2 points good
    df['variant_angles_prob'] = df['keypoint_scores'].apply(lambda x: count_lenghts_prob(x, connections))
