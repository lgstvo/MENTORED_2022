from multiprocessing.sharedctypes import Value
from struct import pack
from unittest import skip
import pyshark
import pandas as pd
import os
import argparse
from dbscan import *

def fetch_package_csv_modified(presaved_packets, window_size, dataset):
    list_csvs = sorted(os.listdir(presaved_packets))
    for packet_file in list_csvs:
        print("Processing {}...".format(packet_file))
        fraction_csv = pd.read_csv(os.path.join(presaved_packets, packet_file))
        entropy_dframe = entropy_dataframe(fraction_csv, window_size, dataset)
        entropy_dframe.to_csv("data/capture45/csvs/entropy/entropy_capture45_{}_window{}.csv".format(packet_file.split('.')[0][-1], window_size))

    print("Fetching complete.")

def compact_entropy_csv(presaved_entropies, window_size):
    list_csvs = sorted(os.listdir(presaved_entropies))
    dframe = pd.DataFrame(columns=['Source_int', 'Destination_int', "Source_Port", "Destination_Port", "Length", "Infected"])
    for packet_file in list_csvs:
        print("Processing {}...".format(packet_file))
        fraction_csv = pd.read_csv(os.path.join(presaved_entropies, packet_file))
        dframe =  pd.concat([dframe, fraction_csv], ignore_index=True, axis=0)

    dframe.to_csv(presaved_entropies+"entropy_capture45_window{}.csv".format(window_size))
    print("Fetching complete.")

def main():
    presaved_packets = "data/capture45/csvs/packets/"
    presaved_entropies = "data/capture45/csvs/entropy/"
    dataset = "ctu13c45"
    window_size = 5000
    fetch_package_csv_modified(presaved_packets, window_size, dataset)
    compact_entropy_csv(presaved_entropies, window_size)
    print("Completed.")

if __name__ == '__main__':
    main()