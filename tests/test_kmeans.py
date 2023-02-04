# Write your k-means unit tests here
import random
import pytest
from sklearn.cluster import KMeans as KMeans_sklearn
import numpy as np
from cluster import KMeans, utils, Silhouette

random_n = np.random.randint(1, 1000)
random_k = np.random.randint(1, random_n)
mat, labels = utils.make_clusters(n=random_n, m=2, k=random_k)

def test_empty():
    """
    Test that kmeans raises a ValueError if data matrix is empty
    """
    with pytest.raises(ValueError):
        mat_, labels_ = utils.make_clusters(n=1000, m=0, k=3)
        kmeans = KMeans(k=3, tol=1e-6, max_iter=100)
        kmeans.fit(mat_)
        kmeans.predict(mat_)

def test_k_not_integer():
    """
    Test if k is an integer
    """
    with pytest.raises(TypeError):
        kmeans = KMeans(k=1e-6, tol=1e-6, max_iter=100)

def test_tol_not_float():
    """
    Test if tol is a float
    """
    with pytest.raises(TypeError):
        kmeans = KMeans(k=3, tol=0, max_iter=100)

def test_max_iter_not_float():
    """
    Test if max_iter is a float
    """
    with pytest.raises(TypeError):
        kmeans = KMeans(k=3, tol=1, max_iter=1e-6)

def test_if_k_is_zero():
    """
    Test if k value is zero
    """
    with pytest.raises(ValueError):
        kmeans = KMeans(k=0, tol=1e-6, max_iter=100)

def test_if_tol_is_zero():
    """
    Test if tol value is zero
    """
    with pytest.raises(ValueError):
        kmeans = KMeans(k=3, tol=0.0, max_iter=100)

def test_if_max_iter_is_zero():
    """
    Test if max_iter value is zero
    """
    with pytest.raises(ValueError):
        kmeans = KMeans(k=3, tol=1e-6, max_iter=0)

def test_correct_number_of_clusters():
    """
    Test if my kmeans implementation returns the right number of clusters at the end of the fit() and predict() methods.
    Compare to sklearn kmeans implementation.
    """
    kmeans = KMeans(random_k, tol=1e-6, max_iter=100)
    kmeans.fit(mat)
    kmeans.predict(mat)

    kmeans_sklearn = KMeans_sklearn(n_clusters=random_k, init='k-means++', n_init='auto', max_iter=100, tol=1e-6)
    kmeans_sklearn.fit(mat)
    kmeans_sklearn.predict(mat)
    assert len(kmeans.get_centroids()) == len(kmeans_sklearn.cluster_centers_)

def test_kmeans_accuracy_against_sklearn_kmeans():
    """
    Test accuracy of kmeans implementation against accuracy of sklearn kmeans at predicting labels
    """
    kmeans = KMeans(random_k, tol=1e-6, max_iter=100)  # Instantiate a kmeans object from the class I wrote
    kmeans.fit(mat)  # Fit kmeans onto the data matrix mat
    my_generated_labels = kmeans.predict(mat)  # Generate labels for each data point

    # Repeat above steps using sklearn.cluster.Kmeans
    kmeans_sklearn = KMeans_sklearn(n_clusters=random_k, init='k-means++', n_init='auto', max_iter=100, tol=1e-6)
    kmeans_sklearn.fit(mat)
    kmeans_sklearn.predict(mat)

    # Labels generated by sklearn Kmeans algorithm
    # (I'm not sure why but label order was reversed when generated by sklearn
    # as compared to labels created by utils.make_clusters)
    sklearn_labels = kmeans_sklearn.labels_[::-1]

    # Calculate what percentage of values are the same in the labels array vs the generated labels
    similarity_percent_kmeans = np.mean(labels == my_generated_labels)
    similarity_percent_sklearn = np.mean(labels == sklearn_labels)

    diff = np.abs((np.mean(similarity_percent_sklearn) - np.mean(similarity_percent_kmeans)))

    # Assert that my_generated_labels are closer to the actual labels
    # ex: for n=1000, m=2, k=3, my similarity percent was 0.937, sklearn's was 0.861
    assert diff < 0.01






