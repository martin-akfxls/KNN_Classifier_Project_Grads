"""
knn_functions.py
Standalone KNN helper functions used by the Pokémon Legendary Classifier.
These demonstrate the core KNN algorithm clearly for academic purposes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ──────────────────────────────────────────────────────────────────────────────
# DISTANCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two feature vectors.

    Formula: d = sqrt( sum( (a_i - b_i)^2 ) )

    Parameters:
        a : numpy array of feature values (e.g. [HP, Atk, Def, ...])
        b : numpy array of feature values
    Returns:
        float : the distance value
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Manhattan distance between two feature vectors.

    Formula: d = sum( |a_i - b_i| )

    Parameters:
        a : numpy array of feature values
        b : numpy array of feature values
    Returns:
        float : the distance value
    """
    return float(np.sum(np.abs(a - b)))


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE SCALING
# ──────────────────────────────────────────────────────────────────────────────

def minmax_normalize(X_train: np.ndarray, X_test: np.ndarray = None):
    """
    Apply Min-Max Normalization to features.

    Formula: x_scaled = (x - min) / (max - min)
    Scales all values to [0, 1] range.

    IMPORTANT: The scaler is fitted on training data only,
    then applied to both training and test data to prevent data leakage.

    Parameters:
        X_train : numpy array of training features
        X_test  : (optional) numpy array of test features
    Returns:
        X_train_scaled, X_test_scaled (or just X_train_scaled if no test)
        scaler : the fitted MinMaxScaler object
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


# ──────────────────────────────────────────────────────────────────────────────
# CORE KNN CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def knn_classify(
    X_train: np.ndarray,
    y_train,
    query: np.ndarray,
    k: int,
    distance_fn=euclidean_distance
) -> tuple:
    """
    Classify a single query record using the KNN algorithm.

    Steps performed:
        1. Compute distance from query to every training record
        2. Sort training records by ascending distance
        3. Select the K nearest neighbors
        4. Use majority vote to decide the class

    Parameters:
        X_train     : 2D numpy array of training features (already scaled)
        y_train     : array-like of training labels
        query       : 1D numpy array of feature values for the new record
        k           : number of nearest neighbors to consider
        distance_fn : distance function to use (default: euclidean)

    Returns:
        predicted_class : str
        neighbors       : list of dicts with keys {rank, index, distance, class}
    """
    if isinstance(y_train, pd.Series):
        y_train_vals = y_train.values
    else:
        y_train_vals = np.array(y_train)

    # Step 1: Compute all distances
    distances = [distance_fn(query, x) for x in X_train]

    # Step 2: Sort by distance (ascending)
    sorted_indices = np.argsort(distances)

    # Step 3: Select K nearest
    k_indices = sorted_indices[:k]

    neighbors = [
        {
            "rank": rank + 1,
            "index": int(idx),
            "distance": round(distances[idx], 5),
            "class": y_train_vals[idx]
        }
        for rank, idx in enumerate(k_indices)
    ]

    # Step 4: Majority vote
    neighbor_classes = [n["class"] for n in neighbors]
    predicted_class = max(set(neighbor_classes), key=neighbor_classes.count)

    return predicted_class, neighbors


def knn_predict_batch(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    k: int,
    distance_fn=euclidean_distance
) -> list:
    """
    Classify all records in X_test using knn_classify().

    Parameters:
        X_train : 2D numpy array (training, already scaled)
        y_train : labels for training set
        X_test  : 2D numpy array (test, already scaled)
        k       : number of neighbors

    Returns:
        List of predicted class labels (one per test record)
    """
    return [knn_classify(X_train, y_train, row, k, distance_fn)[0] for row in X_test]


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(y_true, y_pred, positive_class: str = "Legendary") -> dict:
    """
    Compute a binary confusion matrix manually.

    Returns a dict with keys: TP, TN, FP, FN
    """
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p != positive_class)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def compute_metrics(cm: dict) -> dict:
    """
    Compute accuracy, precision, recall, and F1-score from a confusion matrix dict.

    Formulas:
        Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * (Precision * Recall) / (Precision + Recall)
    """
    TP, TN, FP, FN = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total = TP + TN + FP + FN

    accuracy  = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4)
    }


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def validate_k(k: int, n_train: int) -> str | None:
    """
    Validate the K value.
    Returns an error message string if invalid, None if valid.
    """
    if k <= 0:
        return "K must be a positive integer (> 0)."
    if k > n_train:
        return f"K ({k}) cannot be larger than the training set size ({n_train})."
    return None
