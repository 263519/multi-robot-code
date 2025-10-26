import numpy as np

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
        return np.zeros(2,1)
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

def is_within_bounds(pos: np.ndarray, resolution: float, origin: list, gridmap: np.ndarray) -> bool:
    """
    Functin to check if a robot pose is within the bounds of the gridmap

    Parameters:
        pos (np.ndarray): 2x1 robot pose vector in world coordinates
        resolution (float): resolution of the gridmap
        origin (list): list containing the origin of the gridmap in world coordinates [x, y]
        gridmap (np.ndarray): gridmap of the environment for flocking

    Return:
        bool: result of check
    """
    # Getting bounds of the gridmap
    min_x = origin[0]
    min_y = origin[1]
    max_x = min_x + (gridmap.shape[1] * resolution)
    max_y = min_y + (gridmap.shape[0] * resolution)

    if (min_x < pos[0,0] < max_x) and (min_y < pos[1,0] < max_y):
        return True
    else:
        return False
