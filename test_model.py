import numpy as np
from mfcc_dtw import calculate_dtw_distance

def test_model(patient_pattern, model):
    """
    Test the patient pattern against the trained model.

    Parameters:
    - patient_pattern (array): Preprocessed patient pattern.
    - model (list): Trained model.

    Returns:
    - result_index (int): Index of the closest match in the model.
    """
    # Compare patient pattern with each centroid in the model using DTW
    distances = [calculate_dtw_distance(patient_pattern, centroid) for centroid in model]

    # Get the index of the minimum distance
    min_distance_index = np.argmin(distances)

    return min_distance_index
