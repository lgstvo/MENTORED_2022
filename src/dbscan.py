import pyshark
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from math import floor
import scipy.stats as scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

BOTNET1 = "147.32.84.165"
BOTNET2 = "147.32.84.191"
BOTNET3 = "147.32.84.192"

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

def check_botnet(ip, botnets):
    ip_is_infected = 0
    for infected_host in botnets:
        ip_is_infected = ip == infected_host
        if ip_is_infected:
            return True
    return False

def check_infection(dframe_slice):
    for ip in dframe_slice["Source"]:
        ip_is_infected = check_botnet(ip, [BOTNET1, BOTNET2, BOTNET3])
        if ip_is_infected:
            return True
    return False

def entropy_dataframe(dframe, slice_window):
    # generate new dataframe with values gruped by entropy, to be processed by PCA
    entropy_dframe = pd.DataFrame()

    slices =  list(range(0, len(dframe), slice_window))
    for initial_index in slices:
        ending_index = initial_index+slice_window-1
        dframe_slice = dframe[initial_index : ending_index]
        is_infected = check_infection(dframe_slice)
        entropy_dframe_row = entropy_window(dframe_slice[["Source_int", "Destination_int", "Source_Port", "Destination_Port", "Length"]])
        entropy_dframe_row["Infected"] = is_infected
        entropy_dframe = pd.concat([entropy_dframe, entropy_dframe_row], ignore_index=True, axis=0)

    return entropy_dframe

def generate_PCA(entropy_dframe):
    # creates PCA points for the data
    entropy_dframe_numpy = entropy_dframe.loc[:, entropy_dframe.columns!='Infected'].to_numpy()
    pca = PCA(n_components = 2)
    pca.fit(entropy_dframe_numpy.transpose())
    pca_points = pca.components_
    pca_points = np.vstack([pca_points, entropy_dframe.loc[:, entropy_dframe.columns=='Infected'].to_numpy().transpose()])
    return pca_points

def fetch_package_csv():
    csv1 =  pd.read_csv("packets_csv.csv")
    csv2 =  pd.read_csv("packets_csv2.csv")
    csv3 =  pd.read_csv("packets_csv3.csv")
    full_csv =  pd.concat([csv1, csv2, csv3], ignore_index=True, axis=0)
    full_csv = full_csv.drop(labels="Unnamed: 0", axis=1)
    return full_csv

def generate_plot(filename, points):
    colors = ['blue', 'red']
    print(points)
    plt.scatter(points[0], points[1], c=points[2])
    plt.savefig(filename)

def main(args):
    if args.presaved:
        savefigure_path = "../data/img/pktAll_windowSize{}.png".format(args.slice_window)
        dframe = fetch_package_csv()
    else:
        savefigure_path = "../data/img/pkt{}_windowSize{}.png".format(args.packet_count, args.slice_window)
        dframe = capture_packages(args.packet_count)

    if args.time_cut != -1:
        packets_per_second = floor(len(dframe)/973)
        print(packets_per_second)
        cut_capture = packets_per_second*args.time_cut
        savefigure_path = "../data/img/pkt{}_windowSize{}.png".format(cut_capture, args.slice_window)
        dframe = dframe[0:cut_capture]
    
    entropy_dframe = entropy_dataframe(dframe, args.slice_window)
    points = generate_PCA(entropy_dframe)

    generate_plot(savefigure_path, points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method Configuration")

    parser.add_argument('--packet_count', type=int, default=10000)
    parser.add_argument('--slice_window', type=int, default=50)
    parser.add_argument('--file_name', default='data/capture/original/capture20110818-2.truncated.pcap')
    parser.add_argument('--files_folder', default='.', help="Folder where multiple pcaps are located")
    parser.add_argument('--presaved', default=False, type=bool)
    parser.add_argument('--time_cut', type=int, default=-1)
    args = parser.parse_args()
    main(args)