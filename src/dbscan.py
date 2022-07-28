from cmath import inf
import os
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
# CAPTURE 45
# - Infected hosts
#     - 147.32.84.165: Windows XP (English version) Name: SARUMAN (Label: Botnet) (amount of infected flows: 5160)
# - Normal hosts:
#     - 147.32.84.170 (amount of bidirectional flows: 12133, Label: Normal-V42-Stribrek)
#     - 147.32.84.134 (amount of bidirectional flows: 10382, Label: Normal-V42-Jist)
#     - 147.32.84.164 (amount of bidirectional flows: 2474, Label: Normal-V42-Grill)
#     - 147.32.87.36 (amount of bidirectional flows: 89, Label: CVUT-WebServer. This normal host is not so reliable since is a webserver)
#     - 147.32.80.9 (amount of bidirectional flows: 13, Label: CVUT-DNS-Server. This normal host is not so reliable since is a dns server)
#     - 147.32.87.11 (amount of bidirectional flows: 4, Label: MatLab-Server. This normal host is not so reliable since is a matlab server)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# CAPTURE 51
# - Infected hosts
#     - 147.32.84.165: Windows XP English version Name: SARUMAN. Label: Botnet. Amount of bidirectional flows: 9579
#     - 147.32.84.191: Windows XP English version Name: SARUMAN1. Label: Botnet. Amount of bidirectional flows: 10454
#     - 147.32.84.192: Windows XP English version Name: SARUMAN2. Label: Botnet. Amount of bidirectional flows: 10397
#     - 147.32.84.193: Windows XP English version Name: SARUMAN3. Label: Botnet. Amount of bidirectional flows: 10009
#     - 147.32.84.204: Windows XP English version Name: SARUMAN4. Label: Botnet. Amount of bidirectional flows: 11159
#     - 147.32.84.205: Windows XP English version Name: SARUMAN5. Label: Botnet. Amount of bidirectional flows: 11874
#     - 147.32.84.206: Windows XP English version Name: SARUMAN6. Label: Botnet. Amount of bidirectional flows: 11287
#     - 147.32.84.207: Windows XP English version Name: SARUMAN7. Label: Botnet. Amount of bidirectional flows: 10581
#     - 147.32.84.208: Windows XP English version Name: SARUMAN8. Label: Botnet. Amount of bidirectional flows: 11118
#     - 147.32.84.209: Windows XP English version Name: SARUMAN9. Label: Botnet. Amount of bidirectional flows: 9894
# - Normal hosts:
#     - 147.32.84.170 (amount of bidirectional flows: 10216, Label: Normal-V42-Stribrek)
#     - 147.32.84.134 (amount of bidirectional flows: 1091, Label: Normal-V42-Jist)
#     - 147.32.84.164 (amount of bidirectional flows: 3728, Label: Normal-V42-Grill)
#     - 147.32.87.36 (amount of bidirectional flows: 99, Label: CVUT-WebServer. This normal host is not so reliable since is a webserver)
#     - 147.32.80.9 (amount of bidirectional flows: 651, Label: CVUT-DNS-Server. This normal host is not so reliable since is a dns server)
#     - 147.32.87.11 (amount of bidirectional flows: 4, Label: MatLab-Server. This normal host is not so reliable since is a matlab server)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# CAPTURE 52
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

INFECTED_HOSTS_C45 = ["147.32.84.165"]

INFECTED_HOSTS_C51 = [
    "147.32.84.165",
    "147.32.84.191",
    "147.32.84.192",
    "147.32.84.193",
    "147.32.84.204",
    "147.32.84.205",
    "147.32.84.206",
    "147.32.84.207",
    "147.32.84.208",
    "147.32.84.209"
]

INFECTED_HOSTS_C52 = ["147.32.84.165", "147.32.84.191", "147.32.84.192"]

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

def capture_packages(pcap_file_name, packet_count):
    pk_sh = pyshark.FileCapture(pcap_file_name, only_summaries=True)
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

def entropy_window(dframe_slice, dataset):
    # calls scipy.entropy to calculate the entropy in the slice per usuable column
    entropy_dframe_row = []
    dframe_raw = dframe_slice.to_numpy().transpose()
    for column in dframe_raw:
        # _,counts = np.unique(column, return_counts=True)
        counts = column.astype('float64')
        feature_entropy = scipy.entropy(counts)
        if feature_entropy <= 0:
            feature_entropy = -5
        entropy_dframe_row.append(feature_entropy)

    dframe_model = {
        "Source_int": [entropy_dframe_row[0]],
        "Destination_int": [entropy_dframe_row[1]],
        "Source_Port": [entropy_dframe_row[2]],
        "Destination_Port": [entropy_dframe_row[3]],
        "Length": [entropy_dframe_row[4]]
    }
    entropy_dframe_row = pd.DataFrame(dframe_model)

    infected_hosts = check_infected_hosts(dataset)
    ih_count = 0
    for value in dframe_raw[0]:
        for ih in infected_hosts:
            if value == int(ih.replace(".","")):
                ih_count += 1

    return entropy_dframe_row, ih_count/len(dframe_raw[0])

def check_botnet(ip, botnets):
    ip_is_infected = 0
    for infected_host in botnets:
        ip_is_infected = ip == infected_host
        if ip_is_infected:
            return True
    return False

def check_infection(dframe_slice, infected_hosts):
    for ip in dframe_slice["Source"]:
        ip_is_infected = check_botnet(ip, infected_hosts)
        if ip_is_infected:
            return True
    return False

def check_infected_hosts(dataset):
    if dataset == "ctu13c52":
        infected_hosts = INFECTED_HOSTS_C52
    elif dataset == "ctu13c51":
        infected_hosts = INFECTED_HOSTS_C51
    elif dataset == "ctu13c45":
        infected_hosts = INFECTED_HOSTS_C45
    else:
        infected_hosts = []
    return infected_hosts

def entropy_dataframe(dframe, slice_window, dataset):
    # generate new dataframe with values gruped by entropy, to be processed by PCA
    entropy_dframe = pd.DataFrame()

    slices =  list(range(0, len(dframe), slice_window))
    for initial_index in slices:
        ending_index = initial_index+slice_window-1
        dframe_slice = dframe[initial_index : ending_index]
        infected_hosts = check_infected_hosts(dataset)
        is_infected = check_infection(dframe_slice, infected_hosts)
        entropy_dframe_row, infected_packets_rate = entropy_window(dframe_slice[["Source_int", "Destination_int", "Source_Port", "Destination_Port", "Length"]], dataset)
        entropy_dframe_row["Infected"] = is_infected
        entropy_dframe_row["IH_Rate"] = infected_packets_rate
        entropy_dframe = pd.concat([entropy_dframe, entropy_dframe_row], ignore_index=True, axis=0)

    return entropy_dframe

def generate_PCA(entropy_dframe):
    # creates PCA points for the data
    entropy_dframe_numpy = entropy_dframe.loc[:, entropy_dframe.columns!='Infected']
    entropy_dframe_numpy = entropy_dframe_numpy.loc[:, entropy_dframe_numpy.columns!='IH_Rate'].to_numpy()
    pca = PCA(n_components = 2)
    pca.fit(entropy_dframe_numpy.transpose())
    pca_points = pca.components_
    pca_points = np.vstack([pca_points, entropy_dframe.loc[:, entropy_dframe.columns=='Infected'].to_numpy().transpose()])
    return pca_points

def fetch_package_csv(presaved_packets):
    list_csvs = sorted(os.listdir(presaved_packets))
    dframe = pd.DataFrame(columns=['Protocol', 'Source', 'Source_int', 'Destination', 'Destination_int', 'Length', "Source_Port", "Destination_Port", 'Info'])
    for packet_file in list_csvs:
        print("Processing {}...".format(packet_file))
        fraction_csv = pd.read_csv(os.path.join(presaved_packets, packet_file))

        dframe =  pd.concat([dframe, fraction_csv], ignore_index=True, axis=0)
    dframe = dframe.drop(labels="Unnamed: 0", axis=1)
    print("Fetching complete.")
    return dframe

def generate_plot(filename, points):
    colors = ['blue', 'red']
    plt.scatter(points[0], points[1], c=points[2])
    plt.savefig(filename)

def padronize_inputs(args):

    if args.image_save_folder[-1] != '/':
        args.image_save_folder += '/'

    if args.presaved_entropy != '':
        capture_dataset = args.presaved_entropy.split('/')[1]
        args.presaved_entropy += 'window{}/entropy_{}_window{}.csv'.format(args.slice_window, capture_dataset, args.slice_window)

    return args

def separate_time(pps, start, end, window, entropy_dframe, point_window):
    cut_capture = pps*end//window
    if start == 0:
        start_capture = cut_capture-point_window
        if start_capture < 0:
            start_capture = 0
    else:
        start_capture = pps*start//window
    entropy_dframe = entropy_dframe[start_capture:cut_capture]
    return entropy_dframe

def main(args):
    savefigure_path = args.dataset
    if args.presaved_entropy == '':
        if not args.use_raw:
            dframe = fetch_package_csv(args.presaved_packets)
        else:
            savefigure_path = savefigure_path+"_pkt{}".format(args.packet_count)
            dframe = capture_packages(args.pcap_file_name, args.packet_count)

        entropy_dframe = entropy_dataframe(dframe, args.slice_window, args.dataset)
    else:
        entropy_dframe = pd.read_csv(args.presaved_entropy)

    if args.time_cut != -1:
        savefigure_path = savefigure_path+"_start{}_end{}".format(args.time_startpoint, args.time_cut)
        #pps = len(dframe)//973
        pps = 2460
        entropy_dframe = separate_time(pps, args.time_startpoint, args.time_cut, args.slice_window, entropy_dframe, args.point_window)
    else:
        savefigure_path = savefigure_path+"_pktAll"
        
    savefigure_path = savefigure_path+"_windowSize{}".format(args.slice_window)
    if args.drop_port == True:
        print(entropy_dframe)                  
        entropy_dframe = entropy_dframe.drop(['Destination_Port', 'Source_Port'], axis=1)
        print(entropy_dframe)
        savefigure_path = savefigure_path + "_dropport"
    print(entropy_dframe["Infected"].value_counts())
    points = generate_PCA(entropy_dframe)
    ihrate = entropy_dframe["IH_Rate"].to_numpy()
    mean_ihrate = np.delete(ihrate, np.where(ihrate == 0.0))
    mean_ihrate = sum(mean_ihrate)/len(mean_ihrate)
    print(mean_ihrate)

    if args.new_entropy == True:
        savefigure_path = savefigure_path+"_new_entropy.png"
    else:
        savefigure_path = savefigure_path+".png"
    savefigure_path = args.image_save_folder+savefigure_path
    print("IMAGE SAVED AT: ", savefigure_path)
    generate_plot(savefigure_path, points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method Configuration")

    parser.add_argument('--packet_count', type=int, default=10000)
    parser.add_argument('--use_raw', type=bool, default=False, help="Uses raw pcap for capture")
    parser.add_argument('--slice_window', type=int, default=50)
    parser.add_argument('--pcap_file_name', type=str, default='data/capture/original/capture20110818-2.truncated.pcap')
    parser.add_argument('--presaved_packets', type=str, default="data/capture52/csvs/packets/")
    parser.add_argument('--time_cut', type=int, default=-1, help="In which second should the analisys stop. -1 equals full capture")
    parser.add_argument('--time_startpoint', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ctu13c52", help="Currently supported data: ctu13c52, ctu13c45, ctu13c51")
    parser.add_argument('--presaved_entropy', type=str, default='')
    parser.add_argument('--image_save_folder', type=str, default="data/img/")
    parser.add_argument('--new_entropy', type=bool, default=True)
    parser.add_argument('--drop_port', type=bool, default=False)
    parser.add_argument('--point_window', type=int, default=2000)
    args = parser.parse_args()
    args = padronize_inputs(args)
    main(args)