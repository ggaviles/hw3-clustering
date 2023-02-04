"""
BMI203: Biocomputing algorithms Winter 2022
Assignment 3: KMeans
"""

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)
import sklearn.cluster
from sklearn.cluster import KMeans as KMeans_sklearn

__version__ = "0.1.0"
__author__ = "Giovanni Aviles"