import pandas as pd

file_path = "/dartfs-hpc/rc/lab/M/MHBLD/25_FacebookCOVID/1_Data"
obesity_meta = pd.read_csv(file_path + "220108_facebook_obesity_withindex.csv") 
obesity = pd.read_csv(file_path + "211105_feature_matrix_bert.csv")

obesity["meta_index_map"] = ""
for i in range(0, len(obesity)):
    obesity["meta_index_map"][i] = obesity_meta[obesity_meta['Message'].str.contains(obesity.loc[i, "A"], 
                                                  na = False)].index

obesity.to_csv(file_path + "220109_text_with_meta_map.csv")