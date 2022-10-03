from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering, SpectralClustering
from sklearn_som.som import SOM
import pandas as pd
import numpy as np

class Clustering():
    
    def __init__(self, data_csv, entropy_processed):
        self.entropy_processed = entropy_processed
        
        if entropy_processed == False:
            self.dataset = self.process_entropy(data_csv)
        else:
            self.dataset = data_csv

    def process_entropy(self, data_csv):
        pass

    def load_kmeans(self, n_clusters=2, random_state=0):
        self.clust_alg = KMeans(n_clusters, random_state)
        self.method = "KMeans"

    def load_dbscan(self, eps=0.5, min_samples=100):
        self.clust_alg = DBSCAN(eps, min_samples)
        self.method = "DBSCAN"

    def load_som(self, m=2, n=1, dim=2):
        self.clust_alg = SOM(m, n, dim)
        self.method = "SOM"

    def load_birch(self, n_clusters=2):
        self.clust_alg = Birch(n_clusters)
        self.method = "Birch"

    def load_ward(self, n_clusters=2):
        self.clust_alg = AgglomerativeClustering(n_clusters)
        self.method = "Ward"

    def load_spectral(self, n_clusters=2, random_state=0):
        self.clust_alg = SpectralClustering(n_clusters, random_state)
        self.method = "Spectral"

    def clusterize(self):
        pass