from numpy import full
import pandas as pd

csv1 =  pd.read_csv("packets_csv.csv")
csv2 =  pd.read_csv("packets_csv2.csv")
csv3 =  pd.read_csv("packets_csv3.csv")
full_csv =  pd.concat([csv1, csv2, csv3], ignore_index=True, axis=0)
full_csv = full_csv.drop(labels="Unnamed: 0", axis=1)
print(full_csv.head())
print(len(full_csv))