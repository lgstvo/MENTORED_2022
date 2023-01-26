# dataloader wannabe

import os
import pandas as pd
import numpy as np
import scipy.stats as scipy

from dbscan import INFECTED_HOSTS_C51

INFECTED_HOSTS_CIC = [
    "192.168.50.1",
    "192.168.50.4"
]

def load_c51(folder_path):
    sorted_files = sorted(os.listdir(folder_path))
    entropy_dframe = pd.DataFrame()
    infected_list = []
    for file in sorted_files:
        protocol_dict = {}
        file_path = (folder_path+file)
        file_csv = pd.read_csv(file_path, sep=';')
        
        entropy_features =[]
        cropped_csv = file_csv[['frame.len', '_ws.col.SA', '_ws.col.DA', '_ws.col.PR']]

        infected = 0
        new_protocols = cropped_csv['_ws.col.PR'].unique()
        prot = 1
        for protocol in new_protocols:
            protocol_dict[protocol] = prot
            prot = prot + 1
        
        cropped_csv = cropped_csv.replace({'_ws.col.PR':protocol_dict})
        for (column_name, column_data) in cropped_csv.iteritems():
            
            data = column_data.values.tolist()
            if column_name == '_ws.col.SA' or column_name == '_ws.col.DA':
                for i, value in enumerate(data):
                    if column_name == '_ws.col.SA':
                        if value in INFECTED_HOSTS_C51:
                            infected = 1
                    try:
                        data[i] = int(value.replace('.', ''))
                    except ValueError:
                        data[i] = 0
                        continue
            column_entropy = scipy.entropy(data)
            if np.isnan(column_entropy):
                column_entropy = 0
            entropy_features.append(column_entropy)

        infected_list.append(infected)
        dframe_model = {
            "Source_int":       [entropy_features[1]],
            "Destination_int":  [entropy_features[2]],
            #"Protocol":         [entropy_features[3]],
            "Length":           [entropy_features[0]]

        }
        entropy_dframe_row = pd.DataFrame(dframe_model)
        entropy_dframe = pd.concat([entropy_dframe, entropy_dframe_row], ignore_index=True, axis=0)

    
    print(entropy_dframe.head())
    return entropy_dframe, infected_list

if __name__ == "__main__":
    load_c51("../data/CIC/csvs/")
    #load_c51("../data/capture51/csvs/capture_51_csv_parts/")