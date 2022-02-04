from collections import Counter

import ast
import pandas as pd

file_path = "/dartfs-hpc/rc/lab/M/MHBLD/25_FacebookCOVID/1_Data"
bigrams = pd.read_csv(file_path + "/220127_obesity_with_bigrams.csv")

bigrams["bigrams_list"] = bigrams.V2.apply(lambda row: ast.literal_eval(row))

bigrams = bigrams.groupby('lda_topic_renamed').agg({'bigrams_list': 'sum'})
bigrams = bigrams.bigrams_list.apply(lambda row: Counter(row)).to_frame()
print(bigrams.bigrams_list.apply(lambda row: list(row)[0:10]))