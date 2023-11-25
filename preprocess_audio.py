import numpy as np
import librosa

def preprocess_audio(audio_path):
    """
    Preprocess audio by applying pre-emphasis and extracting MFCC features.

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
    mfccs = librosa.feature.mfcc(emphasized_signal, sr=sr, n_mfcc=13)
    
    return mfccs
