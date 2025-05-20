import pandas as pd
import re


posts_df2 = pd.read_excel('Data/OffreF.xlsx') 



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

posts_df = posts_df2.drop(columns=['missions', 'profil', 'stack_technique'], errors='ignore')  # supprimer anciens
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

posts_df1_cleaned = posts_df1
for col in ['Nom_poste', 'Contrat', 'Entreprise', 'Description']:
    posts_df1_cleaned[col] = posts_df1_cleaned[col].astype(str).str.replace(r'<.*?>', '', regex=True)
    posts_df1_cleaned[col] = posts_df1_cleaned[col].str.replace(r'\s+', ' ', regex=True).str.strip()

posts_df1_cleaned = posts_df1_cleaned.drop(columns=['Salaire'])

posts_df1_cleaned['Contrat'] = posts_df1_cleaned['Contrat'].str.lower()
posts_df1_cleaned['Contrat'] = posts_df1_cleaned['Contrat'].apply(normaliser_contrat)

extraits_base = posts_df1_cleaned['Description'].fillna('').apply(lambda x: extraire_bloc_robuste(x, sections_robustes))
posts_df1_cleaned = pd.concat([posts_df1_cleaned, extraits_base], axis=1)

posts_df1_cleaned['groupe_metier'] = posts_df1_cleaned['Nom_poste'].fillna('').apply(attribuer_groupe_metier)

posts_df1_cleaned['Lieu'] = 'France'



#experience to month 

df_combined_posts = pd.concat([posts_df1_cleaned, posts_df_filtred], axis=0, ignore_index=True)
df_cleaned_combined_posts = df_combined_posts

df_cleaned_combined_posts.drop(columns=['Salaire'], inplace=True)

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