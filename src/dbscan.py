import pyshark
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from generate_csv import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# - Infected hosts
#     - 147.32.84.165: Windows XP English version Name: SARUMAN. Label: Botnet. Amount of bidirectional flows: 4151
#     - 147.32.84.191: Windows XP English version Name: SARUMAN1. Label: Botnet. Amount of bidirectional flows: 4006
#     - 147.32.84.192: Windows XP English version Name: SARUMAN2. Label: Botnet. Amount of bidirectional flows: 7
# - Normal hosts:
#     - 147.32.84.170 (amount of bidirectional flows: 581, Label: Normal-V42-Stribrek)
#     - 147.32.84.134 (amount of bidirectional flows: 11, Label: Normal-V42-Jist)
#     - 147.32.84.164 (amount of bidirectional flows: 2113, Label: Normal-V42-Grill)
#     - 147.32.87.36 (amount of bidirectional flows: 1, Label: CVUT-WebServer. This normal host is not so reliable since is a webserver)
#     - 147.32.80.9 (amount of bidirectional flows: 1, Label: CVUT-DNS-Server. This normal host is not so reliable since is a dns server)
#     - 147.32.87.11 (amount of bidirectional flows: 2, Label: MatLab-Server. This normal host is not so reliable since is a matlab server)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_port(info_str:str):
    infos = info_str.split(' ')

    port_found = 0
    for index, string_value in enumerate(infos):
        if string_value == "→":
            ports = (int(infos[index-1]), int(infos[index+1]))
            port_found = 1
            break

    if port_found == 0:    
        ports = (-1, -1)
    
    return ports

def capture_packages(packet_count):
    pk_sh = pyshark.FileCapture('data/capture/capture20110818-2.truncated.pcap', only_summaries=True)
    pk_sh.load_packets(packet_count=packet_count)
    dframe = pd.DataFrame(columns=['No', 'Time', 'Protocol', 'Source', 'Source_int', 'Destination', 'Destination_int', 'Length', "Source_Port", "Destination_Port", 'Info'])
    
    for index, packet in enumerate(pk_sh):
        ports = get_port(packet.info)
        df_row_packet = pd.DataFrame({
            'No': [packet.no], 
            'Time': [packet.time], 
            'Protocol': [packet.protocol], 
            'Source': [packet.source], 
            'Source_int': [int(packet.source.replace(".",""))], 
            'Destination': [packet.destination], 
            'Destination_int': [int(packet.destination.replace(".",""))], 
            'Length': [int(packet.length)], 
            "Source_Port": [ports[0]], 
            "Destination_Port": [ports[1]], 
            'Info': [packet.info]
            })

        dframe = pd.concat([dframe, df_row_packet], ignore_index=True, axis=0)
    
    print("Number of packages captured: ", len(pk_sh))
    return dframe

def entropy_window(dframe_slice):
    # calls scipy.entropy to calculate the entropy in the slice per usuable column
    entropy_dframe_row = []
    dframe_raw = dframe_slice.to_numpy().transpose()
    for column in dframe_raw:
        _,counts = np.unique(column, return_counts=True)
        feature_entropy = scipy.entropy(counts)
        entropy_dframe_row.append(feature_entropy)

    dframe_model = {
        "Source_int": [entropy_dframe_row[0]],
        "Destination_int": [entropy_dframe_row[1]],
        "Source_Port": [entropy_dframe_row[2]],
        "Destination_Port": [entropy_dframe_row[3]],
        "Length": [entropy_dframe_row[4]]
    }
    entropy_dframe_row = pd.DataFrame(dframe_model)
    
    #print(entropy_dframe_row)
    return entropy_dframe_row

def entropy_dataframe(dframe, slice_window):
    # generate new dataframe with values gruped by entropy, to be processed by PCA
    entropy_dframe = pd.DataFrame()

    slices =  list(range(0, len(dframe), slice_window))
    
    for initial_index in slices:
        ending_index = initial_index+slice_window-1
        dframe_slice = dframe[initial_index : ending_index]
        entropy_dframe_row = entropy_window(dframe_slice[["Source_int", "Destination_int", "Source_Port", "Destination_Port", "Length"]])
        entropy_dframe = pd.concat([entropy_dframe, entropy_dframe_row], ignore_index=True, axis=0)

    print(entropy_dframe.head())
    return entropy_dframe

def generate_PCA(entropy_dframe):
    # creates PCA points for the data
    entropy_dframe_numpy = entropy_dframe.to_numpy()
    pca = PCA(n_components = 2)
    pca.fit(entropy_dframe_numpy.transpose())
    pca_points = pca.components_
    return pca_points

def main(args):
    dframe = capture_packages(args.packet_count)
    entropy_dframe = entropy_dataframe(dframe, args.slice_window)
    points = generate_PCA(entropy_dframe)

    plt.plot(points[0], points[1], 'o')
    plt.savefig("data/img/pkt{}.png".format(args.packet_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method Configuration")

    parser.add_argument('--packet_count', type=int, default=10000)
    parser.add_argument('--slice_window', type=int, default=50)
    parser.add_argument('--file_name', default='data/capture/original/capture20110818-2.truncated.pcap')
    parser.add_argument('--files_folder', default='.', help="Folder where multiple pcaps are located")
    args = parser.parse_args()
    main(args)