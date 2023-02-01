import random

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rv_discrete


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        """
        Check input parameters are of correct type
        """

        # Check that k is an integer
        try:
            val = int(k)
        except ValueError:
            print("Input k value is not an integer. Re-enter a k with integer value.")

        # Check that minimum error tolerance is a float value
        try:
            val = float(tol)
        except ValueError:
            print("Input minimum error tolerance value is not a float. "
                  "Re-enter a minimum error tolerance with float value.")

        # Check that max_iter value is an integer
        try:
            val = int(max_iter)
        except ValueError:
            print("Input maximum iteration value is not an integer. "
                  "Re-enter a maximum iteration input with integer value.")

        """
        Initialize attributes
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter



    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        try:
            mat.ndim == 2
        except TypeError:
            print('Incorrect number of dimensions:' + mat.ndim)







    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """



    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """



    def _generate_centroids(self, mat: np.ndarray, ) -> np.ndarray:
        # Initialize an empty set of centroids
        centroids = {}

        # Generate random x coordinate for initial centroid within range of data points
        random_centroid_x = np.random.uniform(low=np.min(mat[:, 0]), high=np.max(mat[:, 0]))

        # Generate random y coordinate for initial centroid within range of data points
        random_centroid_y = np.random.uniform(low=np.min(mat[:, 1]), high=np.max(mat[:, 1]))

        bounds = (random_centroid_x, random_centroid_y)

        # Create a 2D array with
        random_centroid = np.random.uniform(bounds[0], bounds[1], size=(k, 2))

        # Add the random centroid coordinates to the centroid set
        centroids.add(random_centroid)

        # Calculate the Euclidean distance between the random centroid and the rest of the points
        dist = cdist(random_centroid, mat, 'euclidean')

        # Square the distances between the random centroid and the rest of the points
        square_dist = np.square(dist)

        # Sum the squared the distances
        sum_dist = np.sum(square_dist)

        # Divide each element in the squared distances array by the sum of all the squared distances
        prob_dist = square_dist / sum_dist



        return



