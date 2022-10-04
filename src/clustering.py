from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering, SpectralClustering
from sklearn_som.som import SOM
import scipy.stats as scipy
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class Clustering():
    
    def __init__(self, data_csv):
        self.dataset = data_csv
        self.__defineGT__(["147.32.84.165", "147.32.84.191", "147.32.84.192"])
        self.dataset = self.process_entropy()
        self.__compute_PCA__()
        self.clust_alg = None

    def __feature_select__(self):
        self.dataset = self.dataset[["Source_int", "Destination_int", "Source_Port", "Destination_Port", "Length", "point_id"]]
    
    def __compute_PCA__(self):
        data = self.dataset.to_numpy()
        pca = PCA(n_components = 2)
        pca.fit(data.transpose())
        self.pca_points = pca.components_.transpose()

    def __defineGT__(self, botnets):
        n_points_total = self.dataset["point_id"].values[-1]
        infected_list = []
        for point in range(n_points_total+1):
            infected = 0
            source_ips = self.dataset[self.dataset["point_id"] == point]["Source"]
            for ip in source_ips:
                for bot in botnets:
                    if ip == bot:
                        infected = 1
            infected_list.append(infected)
        self.infected = infected_list

    def process_entropy(self):
        self.__feature_select__()
        entropy_dframe = pd.DataFrame()
        n_points_total = self.dataset["point_id"].values[-1]
        for point in range(n_points_total+1):
            window = self.dataset[self.dataset["point_id"] == point]
            entropy_features = []
            for (column_name, column_data) in window.iteritems():
                if column_name == "point_id":
                    continue
                column_entropy = scipy.entropy(column_data)
                entropy_features.append(column_entropy)
            dframe_model = {
                "Source_int":       [entropy_features[0]],
                "Destination_int":  [entropy_features[1]],
                #"Source_Port":      [entropy_features[2]],
                #"Destination_Port": [entropy_features[3]],
                "Length":           [entropy_features[4]]
            }
            entropy_dframe_row = pd.DataFrame(dframe_model)
            entropy_dframe = pd.concat([entropy_dframe, entropy_dframe_row], ignore_index=True, axis=0)
        
        print(entropy_dframe)
        return entropy_dframe

    def load_kmeans(self, n_clusters=2, random_state=0):
        self.clust_alg = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.method = "KMeans"

    def load_dbscan(self, eps=0.005, min_samples=50):
        self.clust_alg = DBSCAN(eps=eps, min_samples=min_samples)
        self.method = "DBSCAN"

    def load_som(self, m=2, n=1, dim=2):
        self.clust_alg = SOM(m=m, n=n, dim=dim)
        self.method = "SOM"

    def load_birch(self, n_clusters=2):
        self.clust_alg = Birch(n_clusters=n_clusters)
        self.method = "Birch"

    def load_ward(self, n_clusters=2):
        self.clust_alg = AgglomerativeClustering(n_clusters=n_clusters)
        self.method = "Ward"

    def load_spectral(self, n_clusters=2, random_state=0):
        self.clust_alg = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
        self.method = "Spectral"

    def clusterize(self):
        if self.clust_alg == None:
            print("Please define clustering algorithm.")
            return

        clusters = self.clust_alg.fit_predict(self.pca_points)
        plt.scatter(self.pca_points[:, 0], self.pca_points[:, 1], c=clusters)
        plt.savefig("./{}.png".format(self.method))
        plt.close()

    def ground_truth(self):
        plt.scatter(self.pca_points[:, 0], self.pca_points[:, 1], c=self.infected)
        plt.savefig("./GT.png")
        plt.close()