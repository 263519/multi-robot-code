import numpy as np
import math
from math import ceil
import cv2

# ================================
# Helper functions for part one of Project 1
# ================================

def vec_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Function to add two 2D column vectors.

    Parameters:
        a (np.ndarray): 2x1 vector
        b (np.ndarray): 2x1 vector

    Returns:
        np.ndarray: 2x1 vector representing a + b
    """
    return a + b

def vec_sub(a:np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Function to subtract two 2D column vectors.

    Parameters:
        a (np.ndarray): 2x1 vector
        b (np.ndarray): 2x1 vector

    Returns:
        np.ndarray: 2x1 vector representing a - b
    """
    return a-b

def vec_scale(a: np.ndarray, s: float) -> np.ndarray:
    """
    Function to scale a 2D column vector by a scalar.

    Parameters:
        a (np.ndarray): 2x1 vector
        s (float): scalar ratio by which vector should be scaled

    Returns:
        np.ndarray: 2x1 scaled vector
    """
    return a * s

def vec_len(a: np.ndarray) -> float:
    """
    Functin to compute the Euclidean length (magnitude) of a 2D vector.

    Parameter:
        a (np.ndarray): 2x1 vector

    Returns:
        float: vector magnitude
    """
    return float(np.linalg.norm(a))

def vec_norm(a: np.ndarray) -> np.ndarray:
    """
    Function to normalize a 2D column vector (to unit length)

    Parameter:
        a (np.ndarray): 2x1 vector

    Returns:
        np.ndarray: Normalized vecor (2x1)
    """
    magnitude = vec_len(a)
    if magnitude == 0.0:
        return np.zeros((2,1))
    return a / magnitude

def clamp_vec(a: np.ndarray, max_len: float) -> np.ndarray:
    """
    Function to limit the magnitude of a vector to a maximum length (max_len)

    Parameter:
        a (np.ndarray): 2x1 vector to be clamped
        max_len (float): max_len to be used to clamp vector a

    Return:
        np.ndarray: Scaled-down vector 
    """
    magnitude = vec_len(a)
    if magnitude > max_len:
        return vec_scale(a, max_len / magnitude)
    return a

def is_within_bounds(pos: np.ndarray, resolution: float, origin: list, gridmap: np.ndarray, allowance=0.5) -> bool:
    """
    Functin to check if a robot pose is within the bounds of the gridmap

    Parameters:
        pos (np.ndarray): 2x1 robot pose vector in world coordinates
        resolution (float): resolution of the gridmap
        origin (list): list containing the origin of the gridmap in world coordinates [x, y]
        gridmap (np.ndarray): gridmap of the environment for flocking
        allowance(float): 'padding' of the boundaries to prevent collision

    Return:
        bool: result of check
    """
    # Getting bounds of the gridmap
    min_x = origin[0] + allowance
    min_y = origin[1] + allowance
    max_x = min_x + (gridmap.shape[1] * resolution) - (2 * allowance)
    max_y = min_y + (gridmap.shape[1] * resolution) - (2 * allowance)
    if (min_x < pos[0,0] < max_x) and (min_y < pos[1,0] < max_y):
        return True
    else:
        return False

def scale_to_limit(current: np.ndarray, request: np.ndarray, max_value):
    """
        Scales down 'request' vector so that |current + scaled_request| == max_value.

        Parameters:
            current (np.ndarray): current vector 
            request (np.ndarray): requesting vector which results in a value larger that the max if added to current vector

        Returns:
            resulting vector with scaled down request
    """
    a = np.dot(request.reshape(2,), request.reshape(2,))
    b = 2 * np.dot(current.reshape(2,), request.reshape(2,))
    c = np.dot(current.reshape(2,), current.reshape(2,)) - max_value**2

    # Solve quadratic: a*α² + b*α + c = 0
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return current  # shouldn't happen numerically
    sqrt_disc = np.sqrt(discriminant)

    # Two roots; we need the positive α that keeps us in [0, 1]
    alpha1 = (-b + sqrt_disc) / (2 * a)
    alpha2 = (-b - sqrt_disc) / (2 * a)
    alpha = max(0, min(alpha1, alpha2, 1)) if alpha1 < 0 or alpha2 < 0 else min(alpha1, alpha2)
    
    scaled_request = alpha * request
    return current + scaled_request

def world_to_map(pos: np.ndarray, origin: list, resolution: float): 
    """
        Function to convert robot position from world coordinates to gridmap coordinates

        Parameters:
            pos (np.ndarray): [2x1] array of robot pose in world coordinates
            origin (list): list of the map origin in world coordinates - [x,y]
            resolution (float): resolution of gridmap

        Returns:
            Tuple pose in gridmap coordinates - (x, y)
    """
    assert pos.shape == (2,1) # ensure robot pose is a column vector
    mx = int((pos[0,0] - origin[0]) / resolution)
    my = int((pos[1,0] - origin[1]) / resolution)

    # return np.array([mx, my]).reshape(2,1)
    return mx, my

def crop_grid(gridmap: np.ndarray, resolution, map_height, map_width, wall_size=0.2):
    """
        Function to crop walls off the gridmap 

        Parameters:
            gridmap: numpy array of the gridmap of the environment
            resolution (float): map resolution 
            map_height: == map_rows in meters
            map_width: == map_cols in meters
            wall_size (float): approximate size of the wall in meters 

        Return:
            cropped: cropped gridmap
    """

    # Convert pool size to pixels
    map_row = ceil(map_height / resolution)
    map_col = ceil(map_width / resolution)
    wall_size = ceil(wall_size / resolution)

    cropped = gridmap.copy()
    cropped[0:wall_size, :] = 0
    cropped[(map_row-wall_size):, :] = 0
    cropped[:, 0:wall_size] = 0
    cropped[:, (map_col-wall_size):] = 0

    return cropped

def inflate_obstacles(gridmap:np.ndarray, resolution:float, inflate_size=0.5):
    """
    Inflate obstacles in a grid map using OpenCV.

    Parameters:
    -----------
    gridmap : np.ndarray
        Occupancy grid (0 = free, 100 = occupied)
    resolution : float
        Map resolution in meters per cell
    inflate_size : float
        Desired inflation radius in meters

    Returns:
    --------
    inflated_map : np.ndarray
        Inflated occupancy grid (0 = free, 100 = inflated obstacle)
    """
    inflate_pixels = int(np.ceil(inflate_size / resolution))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * inflate_pixels + 1, 2 * inflate_pixels + 1))

    # Convert map to binary (1 where occupied, 0 otherwise)
    binary_map = (gridmap == 100).astype(np.uint8)

    # Inflate obstacles
    inflated = cv2.dilate(binary_map, kernel)

    # Convert back to occupancy values
    inflated = inflated * 100

    return inflated

def yaw_from_quaternion(q) -> float:
    """
    Compute planar yaw (rotation about Z) from geometry_msgs/Quaternion.
    Args:
        q (Quaternion): x, y, z, w
    Returns:
        float: yaw angle in radians (-π … +π)
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw
