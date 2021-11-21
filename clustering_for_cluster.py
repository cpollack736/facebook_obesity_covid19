import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, DBSCAN, Birch, KMeans
from sklearn.metrics import silhouette_score

file_path = "/dartfs-hpc/rc/lab/M/MHBLD/25_FacebookCOVID/1_Data"
dat_obesity = "/211105_feature_matrix_bert.csv"
dat_health = "/211118_feature_matrix_bert_health_comparator.csv"
dat_nonhealth = "/211118_feature_matrix_bert_nonhealth_comparator.csv"

dat_obesity = pd.read_csv(str(file_path + dat_obesity))
dat_health = pd.read_csv(str(file_path + dat_health))
dat_nonhealth = pd.read_csv(str(file_path + dat_nonhealth))

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

dat_health_1 = dat_health.loc[:,"abdominal pain":"youtube channel"]
dat_health_2 = dat_health.loc[:,"neg":"compound"]
dat_health_3 = dat_health.loc[:,"WC":"OtherP"]
dat_health_4 = dat_health.loc[:,"BERT 0":"BERT 767"]
dat_health_fm = pd.concat([dat_health_1,
                            dat_health_2,
                            dat_health_3,
                            dat_health_4],
                            axis = 1)
                      
#api whatsapp : without skip
#neg: compound
#WC: OtherP
#BERT 0 : BERT 767
dat_nonhealth_1 = dat_nonhealth.loc[:,"ago father":"zfat mix"]
dat_nonhealth_2 = dat_nonhealth.loc[:,"neg":"compound"]
dat_nonhealth_3 = dat_nonhealth.loc[:,"WC":"OtherP"]
dat_nonhealth_4 = dat_nonhealth.loc[:,"BERT 0":"BERT 767"]
dat_nonhealth_fm = pd.concat([dat_nonhealth_1,
                            dat_nonhealth_2,
                            dat_nonhealth_3,
                            dat_nonhealth_4],
                            axis = 1)

print("Agglomerative")
model_agg = AgglomerativeClustering()
yhat_agg = model_agg.fit_predict(dat_nonhealth_fm)

print("DBSCAN")
model_db = DBSCAN()
yhat_db = model_db.fit_predict(dat_nonhealth_fm)

print("BIRCH")
model_birch= Birch()
yhat_birch = model_birch.fit_predict(dat_nonhealth_fm)

print("KMeans")
model_kmeans = KMeans()
yhat_kmeans = model_kmeans.fit_predict(dat_nonhealth_fm)

nonhealth_labels = pd.DataFrame({"agglomerate": yhat_agg,
                    "dbscan": yhat_db,
                    "birch": yhat_birch,
                    "kmeans": yhat_kmeans})

print("Silhouette Scores for Nonhealth Comparator")
print("Agglomerative")
print(silhouette_score(dat_nonhealth_fm, nonhealth_labels.agglomerate, metric='euclidean')) #0.358
print("DBSCAN")
print(silhouette_score(dat_nonhealth_fm, nonhealth_labels.dbscan, metric='euclidean')) #0.106
print("BIRCH")
print(silhouette_score(dat_nonhealth_fm, nonhealth_labels.birch, metric='euclidean')) #0.343
print("KMeans")
print(silhouette_score(dat_nonhealth_fm, nonhealth_labels.kmeans, metric='euclidean')) #0.241

print("Agglomerative")
model_agg = AgglomerativeClustering()
yhat_agg = model_agg.fit_predict(dat_health_fm)

print("DBSCAN")
model_db = DBSCAN()
yhat_db = model_db.fit_predict(dat_health_fm)

print("BIRCH")
model_birch= Birch()
yhat_birch = model_birch.fit_predict(dat_health_fm)

print("KMeans")
model_kmeans = KMeans()
yhat_kmeans = model_kmeans.fit_predict(dat_health_fm)

health_labels = pd.DataFrame({"agglomerate": yhat_agg,
                    "dbscan": yhat_db,
                    "birch": yhat_birch,
                    "kmeans": yhat_kmeans})

print("Silhouette Scores for Health Comparator")
print("Agglomerative")
print(silhouette_score(dat_health_fm, health_labels.agglomerate, metric='euclidean')) #0.358
print("DBSCAN")
print(silhouette_score(dat_health_fm, health_labels.dbscan, metric='euclidean')) #0.106
print("BIRCH")
print(silhouette_score(dat_health_fm, health_labels.birch, metric='euclidean')) #0.343
print("KMeans")
print(silhouette_score(dat_health_fm, health_labels.kmeans, metric='euclidean')) #0.241

print("Agglomerative")
model_agg = AgglomerativeClustering()
yhat_agg = model_agg.fit_predict(dat_obesity_fm)

print("DBSCAN")
model_db = DBSCAN()
yhat_db = model_db.fit_predict(dat_obesity_fm)

print("BIRCH")
model_birch= Birch()
yhat_birch = model_birch.fit_predict(dat_obesity_fm)

print("KMeans")
model_kmeans = KMeans()
yhat_kmeans = model_kmeans.fit_predict(dat_obesity_fm)

obesity_labels = pd.DataFrame({"agglomerate": yhat_agg,
                    "dbscan": yhat_db,
                    "birch": yhat_birch,
                    "kmeans": yhat_kmeans})

print("Silhouette Scores for Health Comparator")
print("Agglomerative")
print(silhouette_score(dat_obesity_fm, obesity_labels.agglomerate, metric='euclidean')) #0.358
print("DBSCAN")
print(silhouette_score(dat_obesity_fm, obesity_labels.dbscan, metric='euclidean')) #0.106
print("BIRCH")
print(silhouette_score(dat_obesity_fm, obesity_labels.birch, metric='euclidean')) #0.343
print("KMeans")
print(silhouette_score(dat_obesity_fm, obesity_labels.kmeans, metric='euclidean')) #0.241