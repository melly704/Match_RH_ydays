import pandas as pd
import re
from pymongo import MongoClient
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["matchrh"]
collection = db["offres"]

result = collection.delete_many({
    "$or": [
        {"Nom_poste": None},
        {"Nom_poste": {"$exists": False}},
        {"Nom_poste": ""},
        {"Nom_poste": {"$regex": r"^\s*$"}}  
    ]
})
last_offre = collection.find_one(
    {"offre_id": {"$exists": True}},
    sort=[("offre_id", -1)]
)
max_offre_id = int(last_offre["offre_id"]) if last_offre else 0
next_offre_id = max_offre_id + 1
# Lire le dernier document
dernier_doc = collection.find_one(sort=[("_id", -1)])

# Convertir en DataFrame
posts_df2 = pd.DataFrame([dernier_doc])
colonnes_attendues = [
    "Contrat", "Departement", "Description", "Entreprise", "Experience",
    "experience_mois", "groupe_metier", "Lieu", "missions", "Nom_poste",
    "offre_id", "profil", "stack_technique"
]
for col in colonnes_attendues:
    if col not in posts_df2.columns:
        posts_df2[col] = None



#Contrat 
posts_df2['Contrat'] = posts_df2['Contrat'].str.replace(r'\s+', ' ', regex=True).str.lower()
posts_df2['Contrat'] = posts_df2['Contrat'].str.replace(r'\r\n+', ' ', regex=True)

def normaliser_contrat(contrat):
    if pd.isna(contrat):
        return "Non précisé"
    if "cdd" in contrat and ("apprentissage" in contrat or "professionnalisation" in contrat):
        return "CDD-alternance"
    if "cdd" in contrat or "déterminée" in contrat:
        return "CDD"
    if "cdi" in contrat or "indéterminée" in contrat:
        return "CDI"
    if "intérim" in contrat or "intérimaire" in contrat:
        return "Intérim"
    return "Autre"

posts_df2['Contrat'] = posts_df2['Contrat'].apply(normaliser_contrat)



#Groupe métier 
def attribuer_groupe_metier(poste):
    poste = poste.lower()

    if any(x in poste for x in ['data engineer', 'ingénieur data', 'etl', 'développeur data']):
        return 'Data Engineer'
    elif any(x in poste for x in ['data analyst', 'analyste data']):
        return 'Data Analyst'
    elif any(x in poste for x in ['data scientist', 'machine learning', 'modélisation', 'ia', 'intelligence artificielle']):
        return 'Data Scientist'
    elif any(x in poste for x in ['bi', 'data manager', 'business intelligence', 'reporting', 'lead data']):
        return 'BI / Data Manager'
    elif 'architecte' in poste or 'architect' in poste or 'base de données' in poste:
        return 'Data Architect'
    elif any(x in poste for x in ['data steward', 'data quality']):
        return 'Data Steward'
    elif any(x in poste for x in ['consultant data', 'consultant en data', 'consultant données']):
        return 'Consultant Data'
    elif any(x in poste for x in ['développeur', 'dev', 'concepteur', 'application']):
        return 'Développeur SI / Logiciel'
    elif any(x in poste for x in ['système', 'réseau', 'infrastructure', 'administrateur', 'correspondant informatique']):
        return 'Administrateur Systèmes / Réseaux'
    elif any(x in poste for x in ['cybersécurité', 'sécurité', 'ssi', 'iso 27001', 'risques']):
        return 'Cybersécurité'
    elif 'urbaniste' in poste or 'architecture' in poste or 'systèmes d’information' in poste:
        return 'Architecte SI'
    elif any(x in poste for x in ['technicien', 'support', 'helpdesk', 'assistance']):
        return 'Support / Technicien'
    elif any(x in poste for x in ['chef de projet', 'scrum master', 'product owner', 'pilotage']):
        return 'Chef de projet IT'
    elif any(x in poste for x in ['consultant', 'moa', 'amoa']):
        return 'Consultant SI / MOA'

    else:
        return 'Autre'

#Missions/stack-technique

sections_robustes = {
    'missions': ['vos missions', 'missions', 'responsabilités', 'rôle', 'rôle attendu', 'activités principales'],
    'profil': ['profil', 'nous recherchons', 'candidat idéal', 'compétences attendues', 'profil recherché'],
    'stack_technique': ['stack technique', 'technologies', 'outils', 'environnement technique', 'framework', 'langages', 'logiciels', 'compétences requises']
}
sections_robustes['missions'].extend([
    'vous interviendrez',
    'objectifs',
    'vous serez amené à',
    'en charge de',
    'le travail consiste à',
    'tâches',
    'description et livrables de la prestation'
])
if 'skills' not in sections_robustes['stack_technique']:
    sections_robustes['stack_technique'].append('skills')

def extraire_bloc_robuste(description, sections):
    description = str(description).lower()
    description_clean = re.sub(r'[\n\r\t]+', ' ', description)  
    resultats = {key: None for key in sections.keys()}
    indices = {}

    for key, patterns in sections.items():
        for p in patterns:
            match = re.search(rf"{re.escape(p)}[\s:–\-]*", description_clean)
            if match:
                indices[key] = match.start()
                break

    tri_indices = sorted(indices.items(), key=lambda x: x[1])

    for i, (key, start) in enumerate(tri_indices):
        end = len(description_clean)
        if i + 1 < len(tri_indices):
            end = tri_indices[i + 1][1]
        bloc = description_clean[start:end].strip()
        resultats[key] = bloc

    return pd.Series(resultats)

extraits_robustes = posts_df2['Description'].fillna('').apply(lambda x: extraire_bloc_robuste(x, sections_robustes))

for col in extraits_robustes.columns:
    extraits_robustes[col] = extraits_robustes[col].str.replace(r'\s+', ' ', regex=True).str.strip()

posts_df = posts_df2.drop(columns=['missions', 'profil', 'stack_technique'], errors='ignore') 
posts_df = pd.concat([posts_df, extraits_robustes], axis=1)
def nettoyer_bloc(bloc, patterns):
    if pd.isna(bloc):
        return None
    for pattern in patterns:
        bloc = re.sub(rf"{re.escape(pattern)}[\s:–\-]*", '', bloc, flags=re.IGNORECASE)
    return bloc.strip()

titres_a_supprimer = {
    'missions': ['vos missions', 'missions', 'responsabilités', 'rôle', 'rôle attendu', 'activités principales'],
    'profil': ['profil', 'nous recherchons', 'candidat idéal', 'compétences attendues', 'profil recherché'],
    'stack_technique': ['stack technique', 'technologies', 'outils', 'environnement technique', 'framework', 'langages', 'logiciels', 'compétences requises']
}

for section, titres in titres_a_supprimer.items():
    posts_df[section] = posts_df[section].apply(lambda x: nettoyer_bloc(x, titres))
posts_df_filtred = posts_df[
    ~(posts_df['missions'].isna() & posts_df['profil'].isna() & posts_df['stack_technique'].isna())
]
posts_df_filtred = posts_df_filtred[posts_df_filtred['Description'].fillna('').str.len() > 50]

colonnes_a_nettoyer = ['Description', 'missions', 'profil', 'stack_technique']
for col in colonnes_a_nettoyer:
    posts_df_filtred[col] = posts_df_filtred[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

posts_df_filtred['groupe_metier'] = posts_df_filtred['Nom_poste'].fillna('').apply(attribuer_groupe_metier)


# Nettoyage sup

df_cleaned_combined_posts = posts_df_filtred

def convert_experience_en_mois(texte):
    if pd.isna(texte):
        return None
    texte = str(texte).lower()
    if "débutant" in texte:
        return 0
    match = re.search(r'(\d+)\s*an', texte)
    if match:
        return int(match.group(1)) * 12
    return None

df_cleaned_combined_posts['experience_mois'] = df_cleaned_combined_posts['Experience'].apply(convert_experience_en_mois)
duplicated_desc = df_cleaned_combined_posts.duplicated(subset='Description', keep=False)
condition_to_drop = duplicated_desc & (df_cleaned_combined_posts['Lieu'] == 'France')
df_clean_offres = df_cleaned_combined_posts[~condition_to_drop].reset_index(drop=True)




#extraire departement 

def extraire_departement(lieu):
    if pd.isna(lieu):
        return None
    lieu = str(lieu)
    if lieu.lower() in ["france", "ile-de-france"]:
        return lieu
    match = re.match(r'^(\d+)\s*-\s*', lieu)
    if match:
        return match.group(1)
    return None
df_clean_offres['Departement'] = df_clean_offres['Lieu'].apply(extraire_departement)

profiles_df = pd.read_csv("Data/profile_model.csv").fillna("")
offers_df = df_clean_offres
df_offres = offers_df.copy()
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
df_scores['offre_id'] = df_offres['offre_id']
print(df_scores['offre_id'])

df_scores = df_scores.merge(df_candidats, left_on='candidat_id', right_on='candidat_id', how='left')
df_scores = df_scores.merge(df_offres, on='offre_id', how='left', suffixes=('_candidat', '_offre'))
print(df_scores)
candidats_meilleures_offres = (
    df_scores
    .sort_values(by=['candidat_id', 'score'], ascending=[True, False])
    .groupby('candidat_id')
    .head(10)
)
candidats_meilleures_offres = candidats_meilleures_offres[
    ['candidat_id', 'Profil', 'Points_forts', 'Compétence', 'Expérience', 'Nom_poste','Contrat_offre', 'Description',
       'Experience', 'score']
]
offres_meilleurs_candidats = (
    df_scores
    .sort_values(by=['offre_id', 'score'], ascending=[True, False])
    .groupby('offre_id')
    .head(5)
)

offres_meilleurs_candidats = offres_meilleurs_candidats[
    ['offre_id', 'Nom_poste', 'Contrat_offre', 'Description', 'Experience', 'Entreprise','candidat_id', 'Profil',
       'Compétence', 'Expérience', 'score']
]

print(offres_meilleurs_candidats)


