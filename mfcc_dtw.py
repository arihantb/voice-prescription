import librosa
import numpy as np
from dtw import *

def calculate_dtw_distance(pattern1, pattern2):
    """
    Calculate the Dynamic Time Warping (DTW) distance between two patterns.

    Parameters:
    - pattern1 (array): First pattern.
    - pattern2 (array): Second pattern.

    Returns:
    - distance (float): DTW distance.
    """
    reshaped_pattern1 = pattern1.T.reshape(-1, 1)
    reshaped_pattern2 = pattern2.T.reshape(-1, 1)
    euclidean_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

    d, cost_matrix, acc_cost_matrix, path = dtw(reshaped_pattern1, reshaped_pattern2, dist=euclidean_distance)
    return d

def calculate_mfcc(audio_path):
    """
    Calculate MFCC features from an audio file.

    Parameters:
    - audio_path (str): Path to the audio file.

    Returns:
    - mfccs (array): Extracted MFCC features.
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Apply pre-emphasis
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=emphasized_signal, sr=sr, n_mfcc=13)
    
    return mfccs
