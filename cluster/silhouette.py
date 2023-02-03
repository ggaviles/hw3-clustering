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
        # Calculate matrix of distances from each point to each centroid
        distance_matrix = np.array(([([np.linalg.norm(m - c) ** 2 for c in centroids]) for m in mat]))
        # Make matrix of intra-cluster distances, which will be the distances to closest centroid, which will be minimum
        intra_cluster_dist = np.array([min([np.linalg.norm(m - c) ** 2 for c in centroids]) for m in mat])

        # Sort distance_matrix values by size
        for i in range(len(distance_matrix)):
            distance_matrix[i] = sorted(distance_matrix[i])

        # Create matrix of minimum inter-cluster distance values
        mid_dist = []
        for i in range(len(distance_matrix)):
            mid_dist.append(distance_matrix[i][1])

        min_inter_cluster_dist = mid_dist

        # Iterate through intra cluster and inter cluster distances and return max values
        max_btw_a_and_b = []
        for i, j in zip(intra_cluster_dist, min_inter_cluster_dist):
            max_btw_a_and_b.append(max(i, j))

        silhouette_matrix = (min_inter_cluster_dist - intra_cluster_dist) / max_btw_a_and_b

        return silhouette_matrix
