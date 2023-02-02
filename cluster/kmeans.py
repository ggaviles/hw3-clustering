import random

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict


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

        self.centroids = None

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
        self._generate_init_centroid(mat)

        self._generate_k_centroids(mat)

        curr_error = self._determine_error(mat)


        #max_iter_counter = 1

        # max_iter - 1 because already went through one iteration
        for i in range(self.max_iter-1):
            while error
                self._update_centroids(mat)





            max_iter_counter += 1


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

        # I created a dictionary assigning each point to the centroid for which error is minimum
        from collections import defaultdict
        assignment_dict = defaultdict(list)

        for index, val in enumerate(error):
            min_dist_index = np.argmin(error[index])
            assignment_dict[min_dist_index].append(mat[index])

        for i in range(self.max_iter):

            # Generate k centroids
            self._generate_k_centroids(mat)

            # Calculate the distances between k centroids and each element in mat
            dist = self.distance_from_centroids(mat)

            # Assign centroids to their closest data points (placed into dictionary)
            classification = self._assign_cluster(mat)

            # Generate array of error values
            error_array = self._determine_error(mat)

            #

            while error > self.tol:







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
        return self.centroids

    def _generate_init_centroid(self, mat: np.ndarray):
        self.centroids = []
        initial_index = np.random.choice(range(mat.shape[0]), )
        self.centroids.append(mat[initial_index, :].tolist())

    def distance_from_centroids(self, mat: np.ndarray):
        centroids = self.centroids
        dist = cdist(mat, np.array(self.centroids))

        # Already calculating (minimum) distance between points in mat and closest centroid
        dist_squared = np.array([min([np.linalg.norm(m-c)**2 for c in centroids]) for m in mat])
        self.dist_squared = dist_squared
        return self.dist_squared

    def _distance_from_centroids(self, mat: np.ndarray):
        centroids = self.centroids
        dist = cdist(mat, np.array(self.centroids))

        # Already calculating (minimum) distance between points in mat and closest centroid
        dist_squared = np.array([min([np.linalg.norm(m-c)**2 for c in centroids]) for m in mat])
        self.dist_squared = dist_squared

    def _choose_next_centroid(self, mat: np.ndarray):
        self.probs = self.dist_squared / self.dist_squared.sum()
        self.cumulative_probs = self.probs.cumsum()
        r = np.random.uniform(low=0.0, high=1.0)
        index = np.where(self.cumulative_probs >= r)[0][0]
        return mat[index]

    def _generate_k_centroids(self, mat: np.ndarray):
        self._generate_init_centroid(mat)
        while len(self.centroids) < self.k:
            self._distance_from_centroids(mat)
            self.centroids.append(self._choose_next_centroid(mat))
        self.centroids = np.array(self.centroids)

    def _determine_error(self, mat: np.ndarray) -> np.ndarray:
        centroids = self.get_centroids()
        sum_of_squares_error = np.array([np.square([np.sum((m-c)**2) for c in centroids]) for m in mat])
        return sum_of_squares_error

    def _assign_cluster(self, mat: np.ndarray) -> defaultdict:

        # I created a dictionary assigning each point to the centroid for which error is minimum
        assignment_dict = defaultdict(list)

        error = self._determine_error(mat)

        for index, val in enumerate(error):
            min_dist_index = np.argmin(error[index])
            assignment_dict[min_dist_index].append(mat[index])

        return assignment_dict

    def _update_centroids(self, mat: np.ndarray):
        class_dict = self._assign_cluster(mat)

        mean_dict = defaultdict(list)

        for i in class_dict.keys():
            dict_values = list(class_dict.values())
            dict_values_curr = dict_values[i]
            dict_val_mean = map(np.mean, zip(*dict_values_curr))
            dict_val_mean = list(dict_val_mean)
            mean_dict[i].append(dict_val_mean)

        # Reset self.centroids to be empty
        self.centroids = []
        while len(self.centroids) < self.k:
            for key, value in mean_dict.items():
                index = key
                self.centroids.insert(index, value)
        self.centroids = np.array(self.centroids)
