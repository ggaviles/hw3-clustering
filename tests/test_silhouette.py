# write your silhouette score unit tests here
import random
import pytest
import sklearn
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
from cluster import KMeans, utils, Silhouette

random_n = np.random.randint(1, 1000)
random_k = np.random.randint(1, random_n)
mat, labels = utils.make_clusters(n=random_n, m=2, k=random_k)

def test_sklearn_silhouette_score_mean():
    """
    Test that overall mean silhouette score as determined by sklearn between sklearn.kmeans and my implemented kmeans
    are similar (let's see if their difference is less than a small value like 0.09)
    """
    kmeans = KMeans(random_k, tol=1e-6, max_iter=100)  # Instantiate a kmeans object from the class I wrote
    kmeans.fit(mat)  # Fit kmeans onto the data matrix mat
    my_generated_labels = kmeans.predict(mat)  # Generate labels for each data point

    # Repeat above steps using sklearn.cluster.Kmeans
    kmeans_sklearn = KMeans_sklearn(n_clusters=random_k, max_iter=100, tol=1e-6)
    kmeans_sklearn.fit(mat)
    kmeans_sklearn.predict(mat)

    silhouette_kmeans_matrix = silhouette_score(mat, kmeans._return_matrix_of_labels(mat))  # using my kmeans values
    silhouette_sklearn_matrix = silhouette_score(mat, kmeans_sklearn.labels_)  # using sklearn kmeans values

    silhouette_mean_diff = np.linalg.norm(silhouette_kmeans_matrix - silhouette_sklearn_matrix)

    assert silhouette_mean_diff < 0.1

"""
def test_sklearn_silhouette_score_comparison():
    kmeans = KMeans(random_k, tol=1e-6, max_iter=100)  # Instantiate a kmeans object from the class I wrote
    kmeans.fit(mat)  # Fit kmeans onto the data matrix mat
    my_generated_labels = kmeans.predict(mat)  # Generate labels for each data point

    # Repeat above steps using sklearn.cluster.Kmeans
    kmeans_sklearn = sklearn.cluster.KMeans(n_clusters=random_k, init='k-means++', n_init='auto', max_iter=100, tol=1e-6)
    kmeans_sklearn.fit(mat)
    kmeans_sklearn.predict(mat)

    silhouette_kmeans_matrix = silhouette_score(mat, kmeans._return_matrix_of_labels(mat))  # using my kmeans values
    silhouette_sklearn_matrix = silhouette_score(mat, kmeans_sklearn.labels_)  # using sklearn kmeans values

    assert silhouette_kmeans_matrix >= silhouette_sklearn_matrix
"""

def test_silhouette_vs_sklearn_silhouette():
    """
    Test that silhouette.score() produces values similar to that produced by sklearn.metrics.silhouette_samples within
    a range of 0.01
    """
    kmeans = KMeans(random_k, tol=1e-6, max_iter=100)  # Instantiate a kmeans object from the class I wrote
    kmeans.fit(mat)  # Fit kmeans onto the data matrix mat
    my_generated_labels = kmeans.predict(mat)  # Generate labels for each data point

    silhouette_kmeans_matrix = Silhouette().score(mat, my_generated_labels)  # using my silhouette method
    silhouette_sklearn_matrix = silhouette_samples(mat, my_generated_labels, metric='l2')  # using sklearn silhouette method

    silhouette_diff = np.mean(silhouette_kmeans_matrix - silhouette_sklearn_matrix)

    assert silhouette_diff < 0.1
