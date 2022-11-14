from clustering import Clustering
import pandas as pd

from dataloader import load_c51

#data = pd.read_csv("c52_packets_csv.csv")
data , infected = load_c51("../data/capture51/csvs/capture_51_csv_parts/")
clust = Clustering(data, dataset="capture51", infected=infected)

methods = ["KMeans", "DBSCAN", "SOM", "Birch", "Ward", "Spectral"]

c51_checkpoints={
    "udp_1": 5632,
    "udp_end_1": 6508,
    "udp_2": 6581,
    "udp_end_2": 6772,
    "udp_3": 6798,
    "udp_end_3": 6951,
    "icmp_1": 7153,
    "icmp_end_1": 7949,
}

for method in methods:
    clust.load_method(method)
    clust.clusterize("Full")
    clust.ground_truth("Full")

    for chkpnt, time_stamp in c51_checkpoints.items():
        clust.load_method(method)
        clust.clusterize(chkpnt, t=time_stamp)
        clust.ground_truth(chkpnt, t=time_stamp)
    #clust.clusterize("Breakpoint", t=778)
    #clust.ground_truth("Breakpoint", t=778)
    #clust.clusterize("Attack2", t=778+29)
    #clust.ground_truth("Attack2", t=778+29)
    #clust.clusterize("End", t=903)
    #clust.ground_truth("End", t=903)