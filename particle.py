"""PFA Swarm Oprimization technique
"""
import math
import random

import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score

import PFAClustering
import kmeans
from kmeans import KMeans, calc_sse, calc_sse_divided_by_cluster_size, intra_cluster_distance, inter_cluster_distance
from InitializationCentriods import  InitializationCentrioids
from constants import *


def sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
    return calc_sse(centroids, labels, data)

def quantization_error(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        dist = np.linalg.norm(data[idx] - c, axis=1).sum()
        lenght=len(idx)
        if lenght==0:
            lenght=1
        dist /= lenght
        error += dist
    error /= len(centroids)
    return error



class Particle:
    """[summary]

    """


    def get_score(self, data):
        lables=self._predict(data)
        if self.fitness_function==SSE:
            return sse(self.centroids, lables, data)
        elif self.fitness_function==INTRA_CLUSTER_DISTANCE:
            return kmeans.intra_cluster_distance(self.centroids, lables, data)
        elif self.fitness_function==QUANTIZATION_ERROR:#use the kmeans.QR
            return quantization_error(self.centroids, lables, data)
        elif self.fitness_function==SI:
            si=silhouette_score(data, lables)
            return 1-(si+1)/2
        elif self.fitness_function==DBI:
            return davies_bouldin_score(data, lables)




    def __init__(self,
                 n_cluster: int =2,
                 data: np.ndarray=None,
                 initialization_help: int = 0,
                 solution=None,
                 fitness=None,
                 type=1,
                 fitness_function=0,max_iterations=100):
        self.fitness_function = fitness_function
        if type==2:
            self.centroids = solution
            self.current_score = fitness
        elif type==3:
            self.centroids = solution
        else:
            if initialization_help == PFAClustering.RANDOM_INITIALIZATION:
                index = np.random.choice(list(range(len(data))), n_cluster)
                self.centroids = data[index].copy()
            elif initialization_help==PFAClustering.USE_KMEANS:
                kmeans =sklearn.cluster.KMeans(n_clusters=n_cluster,max_iter=int(max_iterations*.70)
,random_state= random.randint(1,1000))
                kmeans.fit(data)
                self.centroids = kmeans.cluster_centers_.copy()
            elif initialization_help==PFAClustering.KMEANS_PLUSE_PLUSE:
                self.centroids=InitializationCentrioids.plus_plus(data,n_cluster,random_state= random.randint(1,1000))
            elif initialization_help==PFAClustering.HYPRID:
                self.centroids = InitializationCentrioids.plus_plus(data, n_cluster,
                                                                    random_state=random.randint(1, 1000))
            self.position = self.centroids.copy()
            self.current_score = self.get_score(data)
            self.current_sse = calc_sse(self.centroids, self._predict(data), data)









    def _predict(self, data: np.ndarray) -> np.ndarray:
        """Predict new data's cluster using minimum distance to centroid
        """
        distance = self._calc_distance(data)
        cluster = self._assign_cluster(distance)
        return cluster

    def _calc_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance between data and centroids
        """
        distances = []
        for c in self.centroids:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance: np.ndarray) -> np.ndarray:
        """Assign cluster to data based on minimum distance to centroids
        """
        cluster = np.argmin(distance, axis=1)
        return cluster


if __name__ == "__main__":
    pass
