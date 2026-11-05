# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:08:28 2026

@author: Pierre Guéno
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


#%% Lecture des données

with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_speasycda2_1h_19840101_19951231.pkl', 'rb') as file:
    omni_speasycda2_1h_19840101_19951231 = pickle.load(file)

with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_speasycda2_1h_19960101_20071231.pkl', 'rb') as file:
    omni_speasycda2_1h_19960101_20071231 = pickle.load(file)
    
del file
#%% Création matrice mask

from dataset_analysis_functions import compute_mask

M = compute_mask(omni_speasycda2_1h_19840101_19951231) #mask (np.array)
M_omni_speasycda2_1h_19840101_19951231 = pd.DataFrame(M, columns=omni_speasycda2_1h_19840101_19951231.columns, 
                                                      index = omni_speasycda2_1h_19840101_19951231.index) #mask (pd.df)

M = compute_mask(omni_speasycda2_1h_19960101_20071231) #mask (np.array)
M_omni_speasycda2_1h_19960101_20071231   = pd.DataFrame(M, columns=omni_speasycda2_1h_19960101_20071231.columns, 
                                                        index = omni_speasycda2_1h_19960101_20071231.index) #mask (pd.df)

del M
#%% function

def df_to_pypots_array(df):
    """PyPOTS expects X with shape (n_samples, n_steps, n_features).
    A single dataframe is treated here as ONE time-series sample."""
    return df.to_numpy(dtype=float)[np.newaxis, :, :]

#======================================================================
#%% Étape 1 : vérification de l'alignement
# ======================================================================
print("Étape 1 : vérification de l'alignement")

assert omni_speasycda2_1h_19840101_19951231.shape == omni_speasycda2_1h_19960101_20071231.shape, \
    f"Formes différentes : {omni_speasycda2_1h_19840101_19951231.shape} vs {omni_speasycda2_1h_19960101_20071231.shape}"
assert M_omni_speasycda2_1h_19840101_19951231.shape == omni_speasycda2_1h_19840101_19951231.shape \
    and M_omni_speasycda2_1h_19960101_20071231.shape == omni_speasycda2_1h_19960101_20071231.shape, \
    "Les masques doivent avoir la même forme que les données."
assert list(omni_speasycda2_1h_19840101_19951231.columns) == list(omni_speasycda2_1h_19960101_20071231.columns), \
    "Les colonnes diffèrent entre les deux périodes."
assert list(M_omni_speasycda2_1h_19840101_19951231.columns) == list(omni_speasycda2_1h_19840101_19951231.columns) \
    and list(M_omni_speasycda2_1h_19960101_20071231.columns) == list(omni_speasycda2_1h_19960101_20071231.columns), \
    "Les colonnes des masques ne correspondent pas à celles des données."

# Même position dans l'année (mois-jour-heure) pour chaque pas de temps
assert (omni_speasycda2_1h_19840101_19951231.index.strftime("%m-%d %H")
        == omni_speasycda2_1h_19960101_20071231.index.strftime("%m-%d %H")).all(), \
    "Désalignement calendaire entre les deux périodes."
print("  -> OK")


# ======================================================================
#%% Étape 2 : génération des NaN synthétiques (double boucle)
#   i = colonne, j = ligne
#   Si M_omni_..._19840101_19951231 (j, i) == 0  ->  NaN dans carved (j, i)
# ======================================================================
print("Étape 2 : génération des NaN synthétiques")
 
n_colonnes = M_omni_speasycda2_1h_19840101_19951231.shape[1]
n_lignes = M_omni_speasycda2_1h_19840101_19951231.shape[0]

# Copier-coller de l'original au début de la boucle
carvedv1_omni_speasycda2_1h_19960101_20071231 = omni_speasycda2_1h_19960101_20071231.copy()

for i in tqdm(range(n_colonnes), desc="Carving des colonnes", unit="colonne"):
    for j in range(n_lignes):
        if M_omni_speasycda2_1h_19840101_19951231.iat[j, i] == 0:
            carvedv1_omni_speasycda2_1h_19960101_20071231.iat[j, i] = np.nan

del i, j, n_colonnes, n_lignes
# ======================================================================
#%% Étape 3 bis : DataFrame de statut des cellules (mask_df_carvingv1)
# ======================================================================
print("Étape 3 : création de mask_df_carvingv1")


M = compute_mask(carvedv1_omni_speasycda2_1h_19960101_20071231) #mask (np.array)
M_carvedv1_omni_speasycda2_1h_19960101_20071231 = pd.DataFrame(M, columns=carvedv1_omni_speasycda2_1h_19960101_20071231.columns, 
                                                               index=carvedv1_omni_speasycda2_1h_19960101_20071231.index) #mask (pd.df)

def mask_creation(M1, M2, 
                  CODE_MANQUE_NATUREL = -1, CODE_MANQUE_SYNTHETIQUE = -2, CODE_MESURE = 1):
    """
    Crée deux DataFrames de statut des cellules à partir des masques
    avant (M1) et après (M2) carving. Convention des entrées : 1 = mesuré, 0 = manquant.

    Retourne (mask_df_texte, mask_df_entiers) :
      - mask_df_texte   : 'mesure' / 'manque synthétique' / 'manque naturel'
      - mask_df_entiers : 1 (mesure) / 2 (manque synthétique) / 0 (manque naturel)
    """
    with tqdm(total=5, desc="Création des masques", unit="étape") as barre:

        # 1) Vérifications d'alignement
        assert M1.shape == M2.shape, "Les masques avant/après carving n'ont pas la même forme."
        assert M1.index.equals(M2.index), "Les index des masques avant/après carving diffèrent."
        assert list(M1.columns) == list(M2.columns), "Les colonnes des masques avant/après carving diffèrent."
        barre.update(1)

        avant = M1.to_numpy()
        apres = M2.to_numpy()

        # 2) Cohérence : le carving ne peut pas créer de mesure
        assert not ((avant == 0) & (apres == 1)).any(), \
            "Des cellules manquantes avant carving sont mesurées après carving."
        barre.update(1)

        # 3) Statuts
        naturel = (avant == 0)
        synthetique = (avant == 1) & (apres == 0)
        mesure = (avant == 1) & (apres == 1)
        assert (naturel | synthetique | mesure).all(), "Des cellules n'ont reçu aucun statut."
        barre.update(1)

        # 4) DataFrame de libellés
        texte = np.select(
            [naturel, synthetique, mesure],
            ["manque naturel", "manque synthétique", "mesure"],
            default="",
        )
        mask_df_texte = pd.DataFrame(texte, index=M1.index, columns=M1.columns)
        barre.update(1)

        # 5) DataFrame d'entiers
        entiers = np.select(
            [naturel, synthetique, mesure],
            [CODE_MANQUE_NATUREL, CODE_MANQUE_SYNTHETIQUE, CODE_MESURE],
            default=-1,
        ).astype(np.int8)
        mask_df_entiers = pd.DataFrame(entiers, index=M1.index, columns=M1.columns)
        barre.update(1)

    return mask_df_texte, mask_df_entiers


maskchar_carvedv1_omni_speasycda2_1h_19960101_20071231, maskint_carvedv1_omni_speasycda2_1h_19960101_20071231 = mask_creation(M_omni_speasycda2_1h_19960101_20071231, 
                                                         M_carvedv1_omni_speasycda2_1h_19960101_20071231)

del M
#%% Export 

# carvedv1_omni_speasycda2_1h_19960101_20071231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/carvedv1_omni_speasycda2_1h_19960101_20071231.pkl')
# maskchar_carvedv1_omni_speasycda2_1h_19960101_20071231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/maskchar_carvedv1_omni_speasycda2_1h_19960101_20071231.pkl')
# maskint_carvedv1_omni_speasycda2_1h_19960101_20071231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/maskint_carvedv1_omni_speasycda2_1h_19960101_20071231.pkl')

