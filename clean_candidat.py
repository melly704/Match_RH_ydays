import pandas as pd
import re
import string


df_profile_cleaned = pd.read_excel('Data/ProfilF.xlsx') 


#nettoyage des colonnes point-forts,Expérience,Compétence

def nettoyer_texte(texte):
    if pd.isna(texte):
        return ""
    texte = re.sub(r'[\r\n\t]+', ' ', texte)
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

colonnes_a_nettoyer = ['Points_forts', 'Expérience', 'Compétence']

for col in colonnes_a_nettoyer:
    df_profile_cleaned[col] = df_profile_cleaned[col].apply(nettoyer_texte)


#Groupe métier
def attribuer_groupe_metier(profil):
    profil = profil.lower()

    if any(x in profil for x in ['data engineer', 'ingénieur data', 'etl', 'développeur data']):
        return 'Data Engineer'
    elif any(x in profil for x in ['data analyst', 'analyste data']):
        return 'Data Analyst'
    elif any(x in profil for x in ['data scientist', 'machine learning', 'modélisation', 'ia', 'intelligence artificielle']):
        return 'Data Scientist'
    elif any(x in profil for x in ['bi', 'data manager', 'business intelligence', 'reporting', 'lead data']):
        return 'BI / Data Manager'
    elif 'architecte' in profil or 'architect' in profil or 'base de données' in profil:
        return 'Data Architect'
    elif any(x in profil for x in ['data steward', 'data quality']):
        return 'Data Steward'
    elif any(x in profil for x in ['consultant data', 'consultant en data', 'consultant données']):
        return 'Consultant Data'
    elif any(x in profil for x in ['développeur', 'dev', 'concepteur', 'application']):
        return 'Développeur SI / Logiciel'
    elif any(x in profil for x in ['système', 'réseau', 'infrastructure', 'administrateur', 'correspondant informatique']):
        return 'Administrateur Systèmes / Réseaux'
    elif any(x in profil for x in ['cybersécurité', 'sécurité', 'ssi', 'iso 27001', 'risques']):
        return 'Cybersécurité'
    elif 'urbaniste' in profil or 'architecture' in profil or 'systèmes d’information' in profil:
        return 'Architecte SI'
    elif any(x in profil for x in ['technicien', 'support', 'helpdesk', 'assistance']):
        return 'Support / Technicien'
    elif any(x in profil for x in ['chef de projet', 'scrum master', 'product owner', 'pilotage']):
        return 'Chef de projet IT'
    elif any(x in profil for x in ['consultant', 'moa', 'amoa']):
        return 'Consultant SI / MOA'

    else:
        return 'Autre'
        

df_profile_cleaned["Metier_regroupe"] = df_profile_cleaned["Profil"].apply(attribuer_groupe_metier)





#extraction de l'expérience en mois
def extraire_mois_v2(texte):
    texte = str(texte).lower()
    
    # Chercher les années
    match_annee = re.search(r'(\d+)\s*(an|ans)', texte)
    # Chercher les mois
    match_mois = re.search(r'(\d+)\s*mois', texte)
    
    total_mois = 0.0
    if match_annee:
        total_mois += float(match_annee.group(1)) * 12
    if match_mois:
        total_mois += float(match_mois.group(1))
        
    return total_mois if total_mois > 0 else 0.0


df_profile_cleaned['Experience_mois'] = df_profile_cleaned['Expérience'].apply(extraire_mois_v2)


#extraction du département
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
df_profile_cleaned['Departement'] = df_profile_cleaned['Lieu_de_recherche'].apply(extraire_departement)
