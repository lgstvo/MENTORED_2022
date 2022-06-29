from multiprocessing.sharedctypes import Value
from struct import pack
from unittest import skip
import pyshark
import pandas as pd
import os
import argparse
from dbscan import *

def fetch_package_csv_modified():
    csv1 =  pd.read_csv("entropy_capture45_1.csv")
    csv2 =  pd.read_csv("entropy_capture45_2.csv")
    csv3 =  pd.read_csv("entropy_capture45_3.csv")
    csv4 =  pd.read_csv("entropy_capture45_4.csv")
    csv5 =  pd.read_csv("entropy_capture45_5.csv")
    full_csv =  pd.concat([csv1, csv2, csv3, csv4, csv5], ignore_index=True, axis=0)
    full_csv = full_csv.drop(labels="Unnamed: 0", axis=1)
    return full_csv

def main():
    dframe = fetch_package_csv_modified()
    dframe.to_csv("entropy_capture45.csv")

if __name__ == '__main__':
    main()