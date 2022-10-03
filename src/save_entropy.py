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
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.84.170"]
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.96.69"]
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.84.134"]
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.84.164"]
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.87.36"]
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.80.9"]
        fraction_csv = fraction_csv[fraction_csv.Source_int != "147.32.87.11"]
        entropy_dframe = entropy_dataframe(fraction_csv, window_size, dataset)
        entropy_dframe.to_csv("data/{}/csvs/new_entropy/window{}/entropy_{}_{}_window{}.csv".format(CAPTURE, window_size, CAPTURE, packet_file.split('.')[0][-1], window_size))

    print("Fetching complete.")

def compact_entropy_csv(presaved_entropies, window_size):
    list_csvs = sorted(os.listdir(presaved_entropies))
    dframe = pd.DataFrame(columns=['Source_int', 'Destination_int', "Source_Port", "Destination_Port", "Length", "Infected", "IH_Rate", "Mean_S_entropy"])
    for packet_file in list_csvs:
        print("Processing {}...".format(packet_file))
        fraction_csv = pd.read_csv(os.path.join(presaved_entropies, packet_file))
        dframe =  pd.concat([dframe, fraction_csv], ignore_index=True, axis=0)

    dframe.drop(dframe.columns[len(dframe.columns)-1], axis=1, inplace=True)
    dframe.to_csv(presaved_entropies+"entropy_{}_window{}.csv".format(CAPTURE, window_size))
    print("Fetching complete.")

def main(args):
    window_size = args.window_size
    presaved_packets = "data/{}/csvs/packets/".format(CAPTURE)
    presaved_entropies = "data/{}/csvs/new_entropy/window{}/".format(CAPTURE, window_size)
    if not os.path.exists(presaved_entropies):
        os.mkdir(presaved_entropies)

    dataset = "ctu13c51"
    fetch_package_csv_modified(presaved_packets, window_size, dataset)
    compact_entropy_csv(presaved_entropies, window_size)
    print("Completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--capture", type=str, default="capture45")
    parser.add_argument("--window_size", type=int, default=1500)

    args = parser.parse_args()
    CAPTURE = args.capture
    main(args)