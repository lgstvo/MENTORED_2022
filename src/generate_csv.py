from multiprocessing.sharedctypes import Value
from struct import pack
from unittest import skip
import pyshark
import pandas as pd
import os

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

def capture_packages(file_name):
    pk_sh = pyshark.FileCapture(file_name, only_summaries=True)
    pk_sh.load_packets()
    dframe = pd.DataFrame(columns=['Protocol', 'Source', 'Source_int', 'Destination', 'Destination_int', 'Length', "Source_Port", "Destination_Port", 'Info'])
    
    for index, packet in enumerate(pk_sh):
        ports = get_port(packet.info)
        try:
            df_row_packet = pd.DataFrame({
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
        except ValueError:
            continue

        dframe = pd.concat([dframe, df_row_packet], ignore_index=True, axis=0)
    
    pk_sh.close()
    return dframe

def main():
    return 0

if __name__ == "__main__":
    packages_folder = "../data/capture"
    dframe = pd.DataFrame(columns=['Protocol', 'Source', 'Source_int', 'Destination', 'Destination_int', 'Length', "Source_Port", "Destination_Port", 'Info'])
    file_number = 0
    for file in os.listdir(packages_folder):
        if file.endswith(".pcap"):
            if file_number < 55:
                file_number += 1
                continue
            print(file)
            file_path = os.path.join(packages_folder, file)
            dframe_batch = capture_packages(file_path)

            dframe = pd.concat([dframe, dframe_batch], ignore_index=True, axis=0)
            print(len(dframe))
            dframe.to_csv("packets_csv3.csv")
        
    print(dframe.head())