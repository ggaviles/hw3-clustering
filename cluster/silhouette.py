import numpy as np
from scipy.spatial.distance import cdist

from cluster import KMeans


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, mat: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # Check if X is in the right format
        """
        try:
            x.ndim == 2
        except TypeError:
            print('Incorrect number of dimensions:' + x.ndim)

        # Check if Y is in the right format
        try:
            y.ndim == 1
        except TypeError:
            print('Incorrect number of dimensions:' + y.ndim)
        """
        distance_matrix = np.array(([([np.linalg.norm(m - c) ** 2 for c in centroids]) for m in mat]))

        intra_cluster_dist = np.array([min([np.linalg.norm(m - c) ** 2 for c in centroids]) for m in mat])

        for i in range(len(distance_matrix)):
            distance_matrix[i] = sorted(distance_matrix[i])

        mid_dist = []
        for i in range(len(distance_matrix)):
            mid_dist.append(distance_matrix[i][1])

        min_inter_cluster_dist = mid_dist

        silhouette_matrix = (min_inter_cluster_dist - intra_cluster_dist) / min_inter_cluster_dist

        return silhouette_matrix
