from clustering import Clustering
import pandas as pd
import time
from dataloader import load_c51

#data = pd.read_csv("c52_packets_csv.csv")
#data , infected = load_c51("../data/capture51/csvs/capture_51_csv_parts/")
#data = pd.read_csv("CIC_packets_csv.csv")
data , infected = load_c51("../data/CIC/csvs/")
#clust = Clustering(data, dataset="capture51", infected=infected)
clust = Clustering(data, dataset="cic", infected=infected)

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

clust.ground_truth("Full")
for method in methods:
    clust.load_method(method)
    print(method)
    clust.clusterize("Full")
    clust.confusion_m("Full")
'''
    for chkpnt, time_stamp in c51_checkpoints.items():
        t = time.time()
        clust.load_method(method)
        clust.clusterize(chkpnt, t=time_stamp)
        clust.ground_truth(chkpnt, t=time_stamp)
        t_end = time.time()
        print("Elapsed {} in {}: {}".format(chkpnt, method, t_end-t))
    #clust.clusterize("Breakpoint", t=778)
    #clust.ground_truth("Breakpoint", t=778)
    #clust.clusterize("Attack2", t=778+29)
    #clust.ground_truth("Attack2", t=778+29)
    #clust.clusterize("End", t=903)
    #clust.ground_truth("End", t=903)
'''