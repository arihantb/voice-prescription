from fuzzywuzzy import fuzz

def fuzzy_decision_logic(symptom, corpus, threshold=45):
    """
    Make a decision on medication based on fuzzy matching.

    Parameters:
    - symptom (str): Extracted symptom from the patient.
    - corpus (list): List of possible symptoms in the corpus.
    - threshold (int): Fuzzy matching threshold.

    Returns:
    - decision (str): Medication decision based on fuzzy matching.
    """
    # Compare symptom with each entry in the corpus using fuzzy matching
    similarity_scores = [(entry, fuzz.token_sort_ratio(symptom, entry)) for entry in corpus.items()]

    # Filter entries with similarity above the threshold
    similar_entries = [entry for entry, score in similarity_scores if score >= threshold]

    if not similar_entries:
        # If no similar entry found, return a default decision or handle accordingly
        return "No specific medication identified."

    # Return the most similar entry
    return max(similar_entries, key=lambda x: fuzz.token_sort_ratio(symptom, x))
