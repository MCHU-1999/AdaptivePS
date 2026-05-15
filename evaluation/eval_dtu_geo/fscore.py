import numpy as np


def fscore(dist_d2s: np.ndarray, dist_s2d: np.ndarray, threshold: float) -> tuple:
    """
    Compute F-score, Precision, and Recall from two arrays of nearest-neighbor distances.

    Args:
        dist_d2s: (N,) or (N,1) array of distances from reconstructed data points to GT STL.
                  Represents Accuracy / Precision direction.
        dist_s2d: (M,) or (M,1) array of distances from GT STL points to reconstructed data.
                  Represents Completeness / Recall direction.
        threshold: distance threshold in scene units (same units as the point cloud coordinates).

    Returns:
        fscore:    float, harmonic mean of precision and recall
        precision: float, fraction of data points within threshold of GT
        recall:    float, fraction of GT points within threshold of data
    """
    dist_d2s = dist_d2s.ravel()
    dist_s2d = dist_s2d.ravel()

    precision = float((dist_d2s < threshold).mean())
    recall    = float((dist_s2d < threshold).mean())

    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    else:
        f = 0.0

    return f, precision, recall
