# Bring in Modules
import html
import pandas as pd
from sentence_transformers import SentenceTransformer

# Bring in Data
file_path = "/dartfs-hpc/rc/lab/M/MHBLD/25_FacebookCOVID/1_Data"
file_name = "/220119_combined_facebook_data.csv"
liwc_labeled = pd.read_csv(str(file_path + file_name))

# Removing Pet Mentions
possible_pet_tags = ["ADOPTION_SERVICE", "ANIMAL_RESCUE_SERVICE", "ANIMAL_SHELTER", "AQUARIUM", "AQUATIC_PET_STORE",
"DOG_BREEDER", "DOG_DAY_CARE_CENTER", "DOG_PARK", "DOG_TRAINING", "DOG_WALKER", "EQUESTRIAN_FACILITY",
"HORSEBACK_RIDING_SERVICE", "HORSE_TRAINER", "KENNEL", "PET", "PETTING_ZOO", "PET_ADOPTION_SERVICE",
"PET_BREEDER", "PET_CAFE", "PET_GROOMER", "PET_SERVICE", "PET_SITTER", "PET_STORE", "PET_SUPPLIES",
"REPTILE_PET_STORE", "ZOO"]

liwc_labeled = liwc_labeled[-(liwc_labeled["Page Category"].isin(possible_pet_tags))] #522,467 not in pet category

# Text Processing
print("Now processing text!")
liwc_labeled['processed_text'] = liwc_labeled.Message.apply(str) #Change to string
liwc_labeled['processed_text'] = liwc_labeled.processed_text.apply(html.unescape) #Remove HTML escape characters
liwc_labeled['processed_text_bert'] = liwc_labeled['processed_text'] #Create new column for BERT-specific embeddings (don't want to remove additional information)

# Run BERT
print("Now running BERT Embeddings!")
bert_model = SentenceTransformer('bert-base-cased')
liwc_labeled["processed_text_bert"] = liwc_labeled["processed_text_bert"].fillna("")
bert_embeddings = bert_model.encode(list(liwc_labeled["processed_text_bert"]))

# Turning into Dataframe
print("Now converting into DataFrame")
bert_dataframe = pd.DataFrame(bert_embeddings)
bert_dataframe.columns = [f"BERT {x}" for x in range(0, len(bert_dataframe.columns))]

print("Now running feature matrix!")
feature_matrix_bert = pd.concat([liwc_labeled.reset_index(),
                                    bert_dataframe.reset_index()],
                                    axis = 1)
file_name = "/220124_feature_matrix_bert.csv"
feature_matrix_bert.to_csv(str(file_path + file_name))