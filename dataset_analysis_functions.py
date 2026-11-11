import numpy as np
import matplotlib.pyplot as plt

#%% mask, stat, timelag matrix

def _to_float_array(X):
    """Convertit X en tableau 2D de floats (None -> NaN)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X
 
def compute_mask(X):
    """
    Matrice Mask M (n x d) :
        M[t, j] = 1 si X[t, j] est observée
        M[t, j] = 0 si X[t, j] est manquante
    """
    X = _to_float_array(X)
    M = (~np.isnan(X)).astype(int)
    return M
 
 
def compute_stat(X, seuil=3):
    """
    Matrice Stat (Qin et al., 2007) (n x d) :
        0 = grand trou   (valeur manquante dans une suite de NaN de longueur > seuil)
        1 = petit trou   (valeur manquante dans une suite de NaN de longueur <= seuil)
        2 = mesure       (valeur observée)
 
    Paramètres
    ----------
    X : array-like (n, d)
    seuil : int
        Durée maximale (en nombre de pas de temps consécutifs manquants) d'un petit trou.
        À adapter selon la fréquence d'échantillonnage et le domaine d'application.
    """
    columns = X.columns
    X = _to_float_array(X)
    n, d = X.shape
    missing = np.isnan(X)
    stat = np.full((n, d), 2, dtype=int)  # par défaut : mesure
 
    for j in range(d):
        t = 0
        while t < n:
            if missing[t, j]:
                debut = t
                while t < n and missing[t, j]:
                    t += 1  
                longueur = t - debut
                stat[debut:t, j] = 1 if longueur <= seuil else 0
            else:
                t += 1
    return stat, pd.DataFrame(stat, columns=columns)

def compute_timelag(X, T=None):
    """
    Matrice Timelag delta (n x d) : durée écoulée depuis la dernière valeur
    effectivement observée, pour chaque variable.
 
        delta[0, j] = 0
        delta[t, j] = T[t] - T[t-1]                   si M[t-1, j] = 1
        delta[t, j] = T[t] - T[t-1] + delta[t-1, j]   si M[t-1, j] = 0
 
    Paramètres
    ----------
    X : array-like (n, d) 
    T : array-like (n,), optionnel
        Timestamps ordonnés. Par défaut T = [0, 1, ..., n-1].
    """
    columns = X.columns
    X = _to_float_array(X)
    n, d = X.shape
    T = np.arange(n, dtype=float) if T is None else np.asarray(T, dtype=float)
    if T.shape[0] != n:
        raise ValueError("T doit avoir la même longueur que X (n observations).")
 
    M = compute_mask(X)
    delta = np.zeros((n, d), dtype=float)
    for t in range(1, n):
        dt = T[t] - T[t - 1]
        delta[t] = np.where(M[t - 1] == 1, dt, dt + delta[t - 1])
    return delta, pd.DataFrame(delta, columns=columns)



#%% heatmap

import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def heatmap_missing_values(df, df_name = 'OMNI', cmap_ = ['#f2f2f2', '#e74c3c']):
    """
    Affiche une heatmap des valeurs manquantes d'un DataFrame.
    - Variables en colonnes (axe X)
    - Temps / Index en ordonnée (axe Y)
    - Deux couleurs distinctes (Présent vs Manquant)
    - Années extraites de l'index pour l'axe Y
    """
    # 1. Création de la matrice binaire (1 = manquant, 0 = présent)
    missing_matrix = df.isnull().astype(int)
    
    # 2. Définition de deux couleurs bien différentes (ex: Gris très clair et Rouge vif)
    cmap = ListedColormap(cmap_)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 3. Affichage de la matrice avec imshow
    # aspect='auto' permet d'adapter l'affichage aux proportions du DataFrame
    im = ax.imshow(missing_matrix, aspect='auto', cmap=cmap, interpolation='none')
    
    # 4. Configuration de l'axe X (Variables en colonnes)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=10)
    
    # 5. Configuration de l'axe Y (Temps / Index - uniquement les années)
    try:
        # Tente d'extraire l'année si l'index est au format datetime ou parseable
        years = pd.to_datetime(df.index).year
    except Exception:
        # Fallback si l'index est déjà numérique (ex: années directes) ou textuel
        years = df.index
        
    n_rows = len(df)
    # Si le jeu de données est grand, on espace les ticks pour garder la lisibilité
    if n_rows > 25:
        step = max(1, n_rows // 12)
        ticks_idx = np.arange(0, n_rows, step)
        ax.set_yticks(ticks_idx)
        ax.set_yticklabels([years[i] for i in ticks_idx], fontsize=10)
    else:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(years, fontsize=10)
        
    # Titres et étiquettes
    ax.set_xlabel("Variables", fontsize=12, fontweight='bold')
    ax.set_ylabel("Années", fontsize=12, fontweight='bold')
    ax.set_title(f"Heatmap des données manquantes du dataset \n{df_name}", fontsize=14, fontweight='bold', pad=15)
        
    # 6. Ajout d'une légende personnalisée pour les deux couleurs
    legend_elements = [
        Patch(facecolor=cmap_[0], edgecolor='gray', label='Valeur présente'),
        Patch(facecolor=cmap_[1], edgecolor='gray', label='Valeur manquante')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()



def heatmap_mask(df, df_name='OMNI', cmap_ = ["#86E2E1", "#4949AB", "#EEE2E2"] ):
    """
    Affiche une heatmap d'un DataFrame de masque à trois états :
    - -1 : donnée manquante originale (bleu ciel)
    - -2 : donnée manquante synthétique (bleu foncé)
    - +1 : donnée mesurée (gris clair)
    - Variables en colonnes (axe X)
    - Temps / Index en ordonnée (axe Y)
    - Années extraites de l'index pour l'axe Y
    """
    # 1. Matrice des valeurs (-2, -1, +1), converties en indices de couleur 0, 1, 2
    values = df.to_numpy()
    state_order = [-2, -1, 1]  # ordre des couleurs dans le colormap
    color_index = np.zeros_like(values, dtype=int)
    for i, v in enumerate(state_order):
        color_index[values == v] = i

    # 2. Couleurs : bleu foncé (-2), bleu ciel (-1), gris clair (+1)
    cmap = ListedColormap(cmap_)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 7))

    # 3. Affichage de la matrice avec imshow
    ax.imshow(color_index, aspect='auto', cmap=cmap, norm=norm, interpolation='none')

    # 4. Configuration de l'axe X (Variables en colonnes)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=10)

    # 5. Configuration de l'axe Y (années uniquement)
    try:
        years = pd.to_datetime(df.index).year
    except Exception:
        years = df.index

    n_rows = len(df)
    if n_rows > 25:
        step = max(1, n_rows // 12)
        ticks_idx = np.arange(0, n_rows, step)
        ax.set_yticks(ticks_idx)
        ax.set_yticklabels([years[i] for i in ticks_idx], fontsize=10)
    else:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(years, fontsize=10)

    # Titres et étiquettes
    ax.set_xlabel("Variables", fontsize=12, fontweight='bold')
    ax.set_ylabel("Années", fontsize=12, fontweight='bold')
    ax.set_title(f"Heatmap du masque des données manquantes du dataset \n{df_name}",
                 fontsize=14, fontweight='bold', pad=15)

    # 6. Légende personnalisée pour les trois états
    legend_elements = [
        Patch(facecolor=cmap_[1], edgecolor='gray', label='Manquant original (-1)'),
        Patch(facecolor=cmap_[0], edgecolor='gray', label='Manquant synthétique (-2)'),
        Patch(facecolor=cmap_[2], edgecolor='gray', label='Mesuré (+1)'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    
    
#%% statistiques (mean, median, gap)

def stats_missing_values(df):
    """
    Calcule les statistiques des données manquantes (0) et des trous consécutifs
    à partir d'une matrice masque (0 = manquant, 1 = mesure).
    """
    results = []

    for col in df.columns:
        # Dans le masque, 0 = donnée manquante
        mask = (df[col] == 0)
        nb_missing = int(mask.sum())
        pct_missing = float(mask.mean() * 100)

        if nb_missing == 0: 
            results.append({
                'variable': col,
                'Nb données manquantes': 0,
                'Pourcentage données manquantes': 0.0,
                'Nb trous': 0,
                'Moyenne taille trou': 0.0,
                'Mediane taille trou': 0.0,
                'Écart-type taille trou': 0.0
            })
            continue

        # Détection robuste des blocs consécutifs de 0 avec NumPy
        mask_arr = mask.to_numpy()
        padded = np.concatenate(([False], mask_arr, [False]))
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        lengths = ends - starts

        nb_trous = len(lengths)
        if nb_trous > 0:
            mean_size = float(np.mean(lengths))
            median_size = float(np.median(lengths))
            std_size = float(np.std(lengths, ddof=0))  # Écart-type de population
        else:
            mean_size = 0.0
            median_size = 0.0
            std_size = 0.0

        results.append({
            'variable': col,
            'Nb données manquantes': nb_missing,
            'Pourcentage données manquantes': pct_missing,
            'Nb trous': nb_trous,
            'Moyenne taille trou': mean_size,
            'Mediane taille trou': median_size,
            'Écart-type taille trou': std_size
        })

    res_df = pd.DataFrame(results)
    res_df.set_index('variable', inplace=True)
    return res_df

#%% histos

def histo_missing_values(df):
    """
    Pour chaque variable d'un DataFrame de masque (0 = manquant, 1 = présent),
    trace un histogramme individuel de la taille des trous (valeurs 0 consécutives) 
    avec une échelle Y logarithmique.
    """
    for col in df.columns:
        vals = df[col].values
        is_missing = (vals == 0)
        
        # S'il n'y a aucune valeur manquante pour cette colonne, on passe
        if not is_missing.any():
            print(f"[{col}] Aucune donnée manquante, histogramme ignoré.")
            continue
            
        # Calcul de la taille des trous consécutifs
        padded = np.r_[1, vals, 1]
        changes = np.diff(padded)
        starts = np.where(changes == -1)[0]
        ends = np.where(changes == 1)[0]
        gap_sizes = ends - starts
        
        # Création d'une figure individuelle pour chaque variable
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Tracé de l'histogramme (bins entiers adaptés aux tailles de trous)
        bins = np.arange(1, gap_sizes.max() + 2) - 0.5
        ax.hist(gap_sizes, bins=bins, edgecolor='black', color='#3498db', alpha=0.85)
        
        # Application de l'échelle Y logarithmique
        ax.set_yscale('log')
        
        # Titres et étiquettes
        ax.set_title(f"Histogramme des tailles de trous - Variable : {col}", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Taille du trou (nombre de points manquants consécutifs)", fontsize=10)
        ax.set_ylabel("Fréquence (Échelle Log)", fontsize=10)
        
        # Grille pour faciliter la lecture sur échelle log
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
#%% matrice corrélation (par rapport à la matrice masque)

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_selection import r_regression


class ExpoNorm(mcolors.Normalize):
    """Norme exponentielle : etire les valeurs proches de 1.

    v -> (exp(k*v) - 1) / (exp(k) - 1)
    Un ecart |r| 0.8 -> 0.9 devient beaucoup plus visible qu'en echelle lineaire.
    """
    def __init__(self, k=8.0, vmin=0.0, vmax=1.0):
        super().__init__(vmin, vmax)
        self.k = k

    def __call__(self, value, clip=None):
        v, _ = self.process_value(value)
        t = (np.exp(self.k * v) - 1.0) / (np.exp(self.k) - 1.0)
        return np.ma.masked_array(t, mask=np.ma.getmaskarray(v))


def matrice_correlation(df, log_scale=True):
    """
    Calcule la matrice de correlation (Pearson) des variables du df,
    la retourne sous forme de DataFrame et affiche le plot de la demi-matrice.

    Parametre
    ---------
    df : DataFrame binaire (0 = donnee manquante, 1 = donnee mesuree)

    Retour
    ------
    corr_df : DataFrame contenant la matrice de correlation complete

    Parametre optionnel
    -------------------
    log_scale : bool (defaut True)
        Si True, echelle exponentielle sur |correlation| (k=8) : la couleur
        monte tres vite vers les valeurs elevees, ce qui rend un ecart
        0.8 vs 0.9 clairement visible.
        Si False, echelle lineaire (0 -> 1).
    """
    cols = list(df.columns)
    n = len(cols)

    # --- Calcul de la demi-matrice (triangle inferieur) via scikit-learn ---
    corr_df = pd.DataFrame(np.eye(n), index=cols, columns=cols, dtype=float)
    for i in range(n):
        for j in range(i):
            # r_regression(X, y) : coefficient de Pearson de chaque colonne de X avec y
            r = r_regression(df[[cols[i]]], df[cols[j]])[0]
            corr_df.iloc[i, j] = r

    # --- Copie sur l'autre demi-matrice (symetrie) ---
    corr_df = corr_df.where(np.tril(np.ones((n, n), dtype=bool)),
                            corr_df.T)

    # --- Plot : demi-matrice avec degrade sur |correlation| ---
    colors = ["#f7f7f7", "#2166ac", "#08306b"]  # clair -> fonce quand |r| -> 1
    cmap = LinearSegmentedColormap.from_list("pearson_abs", colors)

    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.0 * n + 2))
    # image = |correlation| ; demi-matrice vide masquee (triangle superieur)
    valeurs = corr_df.abs().values.copy()
    masque_vide = np.triu(np.ones((n, n), dtype=bool), k=1)
    valeurs = np.ma.masked_array(valeurs, mask=masque_vide)

    if log_scale:
        im = ax.imshow(valeurs, cmap=cmap, norm=ExpoNorm(k=8))
    else:
        im = ax.imshow(valeurs, cmap=cmap, vmin=0.0, vmax=1.0)

    # Numeros dans chaque case
    for i in range(n):
        for j in range(n):
            if j <= i:  # demi-matrice seulement
                val = corr_df.iloc[i, j]
                couleur_txt = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=couleur_txt, fontsize=9)

    # Noms des variables, inclines a 45 degres
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols)

    fig.colorbar(im, ax=ax, label="|correlation| (Pearson)")
    ax.set_title("Matrice de correlation (demi-matrice)")
    plt.tight_layout()
    plt.show()

    return corr_df

    
    
