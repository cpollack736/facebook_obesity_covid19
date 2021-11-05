# Bring in Modules
import pandas as pd
from sentence_transformers import SentenceTransformer

# Bring in Data
file_path = "/dartfs-hpc/rc/lab/M/MHBLD/25_FacebookCOVID_1_Data"
file_name = "/211105_feature_matrix_no_bert.csv"
liwc_labeled = pd.read_csv(str(file_path + file_name))

# Run BERT
print("Now running BERT Embeddings!")
bert_model = SentenceTransformer('bert-base-cased')
bert_embeddings = bert_model.encode(list(liwc_labeled["processed_text_bert"]))

# Turning into Dataframe
print("Now converting into DataFrame")
bert_dataframe = pd.DataFrame(bert_embeddings)
bert_dataframe.columns = [f"BERT {x}" for x in range(0, len(bert_dataframe.columns))]

print("Now running feature matrix!")
feature_matrix_no_bert = pd.concat([liwc_labeled.reset_index(),
                                    bert_dataframe.reset_index()],
                                    axis = 1)
file_name = "/211105_feature_matrix_bert.csv"
feature_matrix_no_bert.to_csv(str(file_path + file_name))