import pandas as pd
from sklearn.cluster import DBSCAN

file_path = "/dartfs-hpc/rc/lab/M/MHBLD/25_FacebookCOVID/1_Data"
dat_obesity = "/211105_feature_matrix_bert.csv"

dat_obesity = pd.read_csv(str(file_path + dat_obesity))

#api whatsapp : without skip
#neg: compound
#WC: OtherP
#BERT 0 : BERT 767
dat_obesity_1 = dat_obesity.loc[:,"api whatsapp":"without skip"]
dat_obesity_2 = dat_obesity.loc[:,"neg":"compound"]
dat_obesity_3 = dat_obesity.loc[:,"WC":"OtherP"]
dat_obesity_4 = dat_obesity.loc[:,"BERT 0":"BERT 767"]
dat_obesity_fm = pd.concat([dat_obesity_1,
                            dat_obesity_2,
                            dat_obesity_3,
                            dat_obesity_4],
                            axis = 1)
dat_obesity_fm.head()                    

model_db = DBSCAN()
yhat_db = model_db.fit_predict(dat_obesity_fm)

dbscan_labels = pd.DataFrame({"dbscan": yhat_db})

dbscan_labels.to_csv("221106_dbscan_labels.csv")