#%% librairies

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


#%% Lecture des données

with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_speasycda2_1h_19840101_19951231.pkl', 'rb') as file:
    omni_speasycda2_1h_19840101_19951231 = pickle.load(file)

with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_speasycda2_1h_19960101_20071231.pkl', 'rb') as file:
    omni_speasycda2_1h_19960101_20071231 = pickle.load(file)
    
with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_speasycda2_1h_20080101_20191231.pkl', 'rb') as file:
    omni_speasycda2_1h_20080101_20191231 = pickle.load(file)
    
del file

#%% Fonctions 

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



#%% Carving

def carving(df1, df2):
    """
    création des données manquantes synthétiques sur le df2 en prenant la distribution du df1. 
    /!\ les df doivent avoir la même forme et taille (colonnes, résolution et période). 
    --> la 1ère étape de la fonction teste cela.

    Parameters
    ----------
    df1 : le df avec la distribution de trous que l'on veut copier
    df2 : le df avec lequel on veut ajouter des données manquantes synthétiques
    

    Returns
    -------
    carved_df2
    maskchar_carved_df2
    maskint_carved_df2
    """
    #%% 1. Création des matrices M
    
    print("Création des matrices M")
    
    from dataset_analysis_functions import compute_mask
    
    M_df1 = pd.DataFrame(compute_mask(df1), columns=df1.columns, index = df1.index)
    M_df2 = pd.DataFrame(compute_mask(df2), columns=df2.columns, index = df2.index)
    
    #%% 2. Vérification de l'alignement des deux df
    
    print("Vérification de l'alignement")
    
    assert df1.shape == df2.shape, \
        f"Formes différentes : {df1.shape} vs {df2.shape}"
    assert M_df1.shape == df1.shape \
        and M_df2.shape == df2.shape, \
        "Les masques doivent avoir la même forme que les données."
    assert list(df1.columns) == list(df2.columns), \
        "Les colonnes diffèrent entre les deux périodes."
    assert list(M_df1.columns) == list(df1.columns) \
        and list(M_df2.columns) == list(df2.columns), \
        "Les colonnes des masques ne correspondent pas à celles des données."

    # Même position dans l'année (mois-jour-heure) pour chaque pas de temps
    assert (df1.index.strftime("%m-%d %H")
            == df2.index.strftime("%m-%d %H")).all(), \
        "Désalignement calendaire entre les deux périodes."
    
    #enchaînement de filtres pour vérifier la même forme, avec le même nombre de données, même colonnes, etc.
    #continue sans provoquer d'erreurs si c'est le cas
    #%% 3. Génération des NaN synthétiques (double boucle)
    #   i = colonne, j = ligne
    #   Si M_omni_..._19840101_19951231 (j, i) == 0  ->  NaN dans carved (j, i)
     
    n_colonnes = M_df1.shape[1]
    n_lignes = M_df1.shape[0]

    # Copier-coller de l'original au début de la boucle
    carved_df2 = df2.copy()

    for i in tqdm(range(n_colonnes), desc="Carving des colonnes", unit="colonne"):
        for j in range(n_lignes):
            if M_df1.iat[j, i] == 0:  #donc qu'une donnée est manquante dans le df1
                carved_df2.iat[j, i] = np.nan  #on créé la donnée manquante synthétique
                

    #%% 4. Création du mask (pour distinguer les données manquantes réelles et synéthiques)

    M_carved_df2 = pd.DataFrame(compute_mask(carved_df2), columns=carved_df2.columns, index=carved_df2.index)
    
    maskchar_carved_df2, maskint_carved_df2 = mask_creation(M_df2, M_carved_df2, CODE_MANQUE_NATUREL = -1, CODE_MANQUE_SYNTHETIQUE = -2, CODE_MESURE = 1)
    #le premier df est composé de char 'mesure' / 'manque synthétique' / 'manque naturel'
    #le second est composé de int (par défaut) : 1 (mesure) / 2 (manque synthétique) / 0 (manque naturel)

    return carved_df2, maskchar_carved_df2, maskint_carved_df2


#%%%

carvedv2_omni_speasycda2_1h_20080101_20191231, maskchar_carvedv2_omni_speasycda2_1h_20080101_20191231, maskint_carvedv2_omni_speasycda2_1h_20080101_20191231 = carving(
    omni_speasycda2_1h_19960101_20071231, #la distribution que l'on regarde
    omni_speasycda2_1h_20080101_20191231) #la période sur laquelle on copie la distribution


# CONVENTIONS NOMS
#%% Export 

#NAME CONVENTION
#v1 -> distribution of SC22(1984-1995)
#v2 -> distribution of SC23(1996-2007)

# carvedv1_omni_speasycda2_1h_19960101_20071231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/carvedv1_omni_speasycda2_1h_19960101_20071231.pkl')
# maskchar_carvedv1_omni_speasycda2_1h_19960101_20071231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/maskchar_carvedv1_omni_speasycda2_1h_19960101_20071231.pkl')
# maskint_carvedv1_omni_speasycda2_1h_19960101_20071231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/maskint_carvedv1_omni_speasycda2_1h_19960101_20071231.pkl')

# carvedv2_omni_speasycda2_1h_20080101_20191231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/carvedv2_omni_speasycda2_1h_20080101_20191231.pkl')
# maskchar_carvedv2_omni_speasycda2_1h_20080101_20191231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/maskchar_carvedv2_omni_speasycda2_1h_20080101_20191231.pkl')
# maskint_carvedv2_omni_speasycda2_1h_20080101_20191231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/maskint_carvedv2_omni_speasycda2_1h_20080101_20191231.pkl')

