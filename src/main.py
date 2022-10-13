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
    clust.clusterize("1min", t=778+60)
    clust.ground_truth("1min", t=778+60)