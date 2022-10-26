from clustering import Clustering
import pandas as pd

data = pd.read_csv("c52_packets_csv.csv")
clust = Clustering(data)

methods = ["KMeans", "DBSCAN", "SOM", "Birch", "Ward", "Spectral"]

for method in methods:
    clust.load_method(method)
    clust.clusterize("Full")
    clust.ground_truth("Full")
    clust.clusterize("Breakpoint", t=778)
    clust.ground_truth("Breakpoint", t=778)
    clust.clusterize("Attack2", t=778+29)
    clust.ground_truth("Attack2", t=778+29)
    clust.clusterize("End", t=903)
    clust.ground_truth("End", t=903)