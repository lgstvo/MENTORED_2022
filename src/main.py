from clustering import Clustering
import pandas as pd

data = pd.read_csv("c52_packets_csv.csv")
clust = Clustering(data)
clust.load_kmeans()
clust.clusterize()
clust.load_dbscan()
clust.clusterize()
clust.load_som()
clust.clusterize()
clust.load_birch()
clust.clusterize()
clust.load_ward()
clust.clusterize()
clust.load_spectral()
clust.clusterize()