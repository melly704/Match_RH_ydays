import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

profiles_df = pd.read_csv("Data/profile_model.csv").fillna("")
offers_df = pd.read_csv("Data/offres_model.csv").fillna("")
df_offres = offers_df.copy()
df_offres['offre_id'] = df_offres.index  
df_candidats = profiles_df.copy()
df_candidats['candidat_id'] = df_candidats.index  
weights = {
    "Metier_regroupe": 0.20,
    "Points_forts": 0.15,
    "Compétence": 0.15,
    "Contrat": 0.15,
    "Experience_mois": 0.20,
    "Departement": 0.15
}
field_map = {
    "Metier_regroupe": "groupe_metier",
    "Points_forts": "stack_technique",
    "Compétence": "stack_technique",
    "Contrat": "Contrat",
    "Experience_mois": "experience_mois",
    "Departement": "Departement",
}
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
similarity_total = np.zeros((len(profiles_df), len(offers_df)))
for profile_field, weight in weights.items():
    
    offer_field = field_map[profile_field]
    profile_texts = profiles_df[profile_field].astype(str).tolist()
    offer_texts = offers_df[offer_field].astype(str).tolist()

    profile_embeddings = model.encode(profile_texts, show_progress_bar=True)
    offer_embeddings = model.encode(offer_texts, show_progress_bar=True)

    sim_matrix = cosine_similarity(profile_embeddings, offer_embeddings)

    similarity_total += weight * sim_matrix

all_matches = []

for i in range(similarity_total.shape[0]):
    for j in range(similarity_total.shape[1]):
        all_matches.append((i, j, similarity_total[i, j]))

df_scores = pd.DataFrame(all_matches, columns=["candidat_id", "offre_id", "score"])
df_scores = df_scores.merge(df_candidats, left_on='candidat_id', right_on='candidat_id', how='left')
df_scores = df_scores.merge(df_offres, on='offre_id', how='left', suffixes=('_candidat', '_offre'))

candidats_meilleures_offres = (
    df_scores
    .sort_values(by=['candidat_id', 'score'], ascending=[True, False])
    .groupby('candidat_id')
    .head(10)
)
candidats_meilleures_offres = candidats_meilleures_offres[
    ['candidat_id', 'Profil', 'Points_forts', 'Compétence', 'Expérience', 'Nom_poste','Contrat_offre', 'Description',
       'Experience', 'Entreprise', 'score']
]
offres_meilleurs_candidats = (
    df_scores
    .sort_values(by=['offre_id', 'score'], ascending=[True, False])
    .groupby('offre_id')
    .head(5)
)
