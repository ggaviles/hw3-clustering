import random

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from matplotlib import pyplot as plt


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
        if k != int(k):
            raise TypeError("Input k value is not an integer. Re-enter a k with integer value.")

        # Check that minimum error tolerance is a float value
        if type(tol) is not float:
            raise TypeError("Input minimum error tolerance value is not a float. "
                  "Re-enter a minimum error tolerance with float value.")

        # Check that max_iter value is an integer
        if max_iter != int(max_iter):
            raise TypeError("Input maximum iteration value is not an integer. "
                  "Re-enter a maximum iteration input with integer value.")

        """
        Check input parameters are greater than zero.
        """

        # Check that k is not zero
        if k == 0:
            raise ValueError("There must be at least one cluster. Set k integer value greater than 0.")

        # Check that tol is not zero
        if tol == 0.0:
            raise ValueError("Tolerance value should be set low but greater than zero. Set tol float value greater than 0.")

        # Check that max_iter is not zero
        if max_iter == 0:
            raise ValueError("There must be at least one iteration. Set max_iter integer value greater than 0.")

        """
        Initialize attributes
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = None
        self.mean_squared_error = None

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
        if mat.size == 0:
            raise ValueError("Array is empty. Enter a 2D matrix with elements.")

        # Create an initial centroid
        self._generate_init_centroid(mat)

        # Generate k centroids
        self._generate_k_centroids(mat)

        # Calculate the sum of squares error for this first set of k centroids
        prev_error = self._generate_error_per_centroid(mat)

        # No other error here so set error_diff to prev_error
        curr_error = prev_error

        error_diff_dict = {}

        for key in curr_error:
            error_diff_dict[key] = np.absolute((prev_error[key])[0] - (curr_error[key])[0])

        # max_iter - 1 because already went through one iteration
        for i in range(self.max_iter-1):
            for values in error_diff_dict.values():
                if float(values) > self.tol:
                    self._update_centroids(mat)
                else:
                    self.mean_squared_error = float(sum(error_diff_dict.values())) / float(len(error_diff_dict.values()))
                    break

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

        # Running self._assign_cluster(mat) would return a dictionary with keys as labels, values as data points
        # new_assign = self._assign_cluster(mat)

        """
        Code below plots points colored by their label:
        
        for key, data_list in new_assign.items():
            key, values = zip(*data_list)  # Unpack
            plt.scatter(key, values, label=key)
        plt.show()
        """

        cluster_labels = self._return_matrix_of_labels(mat)

        return cluster_labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.mean_squared_error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids

    def _generate_init_centroid(self, mat: np.ndarray):
        self.centroids = []  # Set self.centroids variable to be an empty list
        initial_index = np.random.choice(range(mat.shape[0]), )  # Choose a random index in the range of the dataset
        self.centroids.append(mat[initial_index, :].tolist())  # Find data point by its index and add to centroids list

    def distance_from_centroids(self, mat: np.ndarray):
        centroids = self.centroids
        # Calculate squared minimum distance between points in mat and centroids to find distance to closest centroid
        dist_squared = np.array([min([np.linalg.norm(m-c)**2 for c in centroids]) for m in mat])
        self.dist_squared = dist_squared  # Update self.dist_squared variable
        return self.dist_squared

    def _distance_from_centroids(self, mat: np.ndarray):
        centroids = self.centroids
        # Calculate squared minimum distance between points in mat and closest centroid
        dist_squared = np.array([min([np.linalg.norm(m-c)**2 for c in centroids]) for m in mat])
        self.dist_squared = dist_squared  # Update self.dist_squared variable

    def _choose_next_centroid(self, mat: np.ndarray):
        # Divide squared distances by sum of squared distances to get a probability distribution
        self.probs = self.dist_squared / self.dist_squared.sum()
        self.cumulative_probs = self.probs.cumsum()  # Calculate the cumulative sum of the probability distribution
        r = np.random.uniform(low=0.0, high=1.0)  # Choose a uniform random number between 0.0 and 1.0
        # Choose an index for which the cumulative probability is greater than the random number generated
        index = np.where(self.cumulative_probs >= r)[0][0]
        return mat[index]  #

    def _generate_k_centroids(self, mat: np.ndarray):
        self._generate_init_centroid(mat)  # Call method to create random initial centroid
        while len(self.centroids) < self.k:  # Generate k number of centroids
            self._distance_from_centroids(mat)
            self.centroids.append(self._choose_next_centroid(mat))
        self.centroids = np.array(self.centroids)

    def _determine_error(self, mat: np.ndarray) -> np.ndarray:
        centroids = self.get_centroids()
        sum_of_squares_error = np.array([np.square([np.sum((m-c)**2) for c in centroids]) for m in mat])
        return sum_of_squares_error

    def _assign_cluster(self, mat: np.ndarray) -> defaultdict:

        # Create dictionary assigning each point to the centroid for which error is minimum
        assignment_dict = defaultdict(list)
        error = self._determine_error(mat)
        for index, val in enumerate(error):
            min_dist_index = np.argmin(error[index])
            assignment_dict[min_dist_index].append(mat[index])
        return assignment_dict  # Return a dictionary with cluster assignments as keys, corresponding points as values

    def _update_centroids(self, mat: np.ndarray):
        # Creates a dictionary in which keys are centroids and values are the data points assigned to them
        class_dict = self._assign_cluster(mat)

        # Initialize a dictionary with mean coordinates for each cluster center
        mean_dict = defaultdict(list)

        # Iterate through the k keys
        for i in class_dict.keys():
            # Pass the values of class_dict into a list and assign to dict_values
            dict_values = list(class_dict.values())

            # Select the values corresponding to the ith key
            dict_values_curr = dict_values[i]

            # Take the mean of all the n observations and m features across each data point in a particular cluster
            dict_val_mean = map(np.mean, zip(*dict_values_curr))

            # Make dict_val_mean a list
            dict_val_mean = list(dict_val_mean)

            # Append mean_dict with the mean associated with the m and n values
            mean_dict[i].append(dict_val_mean)

        # Reset self.centroids to be empty
        self.centroids = []

        # Loop through the number of centroids until you have added k centroids
        while len(self.centroids) < self.k:

            # Iterate through mean_dict
            for key, value in mean_dict.items():
                # Assign the k key values to be indices for the self.centroids array
                index = key
                # Insert the associated mean values to a specific index
                self.centroids.insert(index, value)

        # Reshape self.centroids array to be m x n dimensions
        self.centroids = np.squeeze(np.array(self.centroids), axis=(1,))

    def _generate_error_per_centroid(self, mat: np.ndarray):
        # Returns dictionary of summed errors per cluster
        error_dict = defaultdict(list)
        error = self._determine_error(mat)
        for index, val in enumerate(error):
            min_dist_index = np.argmin(error[index])
            error_dict[min_dist_index].append(val[min_dist_index])

        # Calculate error of all assigned points in a particular cluster
        sum_errors_dict = {k: [sum(error_dict[k])] for k in error_dict.keys()}

        return sum_errors_dict

    def _return_matrix_of_labels(self, mat: np.ndarray) -> np.ndarray:
        labels = []
        error = self._determine_error(mat)
        for index, val in enumerate(error):
            min_dist_index = np.argmin(error[index])
            labels.append(min_dist_index)
        labels = np.array(labels)
        return labels
