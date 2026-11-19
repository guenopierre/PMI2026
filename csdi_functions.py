import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

# random carving 
def get_rand_mask(observed_mask):
    """Masquage aléatoire pour l'entraînement. Prend le masque des vraies observations et masque une proportion aléatoire pour créer 
    le masque conditionnel (m^co) qui sert pour l'apprentissage

    Args:
        observed_mask (torch.tensor (B,K,L)): matrice masque des vraies observations
    
    Return :
        torch.tensor(B,K,L) : matrice conditionnelle m^co
    """
    # On génère nombres aléatoires sur la grille de observed_mask. En multipliant cela donne un tensor avec 0 les points non observés et nombre aléatoire
    #pour les points observés
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1) # converti en vecteur 1D
    sample_ratio = np.random.rand()  # tirage aléatoire entre 0 et 1 = missin rate
    num_observed = observed_mask.sum().item() #compte le nombre total de points observés
    num_masked = round(num_observed * sample_ratio) #compte le nombre de points à masquer artificiellement
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1 #on remplace les points par -1, sont les cibles d'imputation m^ta
    
    #remplace les valeurs non observés (0) et masqués (-1) par False et les valeurs observés(>0) par True. Redimensionne (B,K,L) et converti en float
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float() 
    
    return cond_mask


#Préparation des données 
def assemble_input_for_training(X_ori, indicating_mask, cond_mask, observed_tp):
    """
    Fonction de préparation des données pour l'entraînement. 
    Extrait les tenseurs d'un batch, adapte leurs dimensions au réseau CSDI et construit le dictionnaire d'entrées

    Args:
        X_ori (): matrice des valeurs temporelles d'origine
        indicating_mask (): masque binaire des cibles d'imputation (m^ta)
        cond_mask (): masque binaire des observations conditionnelles (m^co)
        observed_tp (): les points des temps observés
    
    Return:
        Dict 
    """
    # permute (B,L,K) à (B,K,L) pour le réseau de débruitage
    inputs = {"X_ori" : X_ori.permute(0,2,1), 
              "indicating_mask" : indicating_mask.permute(0,2,1),
              "cond_mask" : cond_mask.permute(0,2,1),
              "observed_tp" : observed_tp}
    return inputs
    
def assemble_input_for_testing(X, cond_mask, observed_tp):
    """
    Fonction de préparation des données pour la test. Permet d'adapter aux dimension du réseau CSDI et construit un dictionnaire

    Args:
        X (_type_): _description_
        cond_mask (_type_): _description_
        observed_tp (_type_): _description_
    """
    inputs = {"X" : X.permute(0,2,1),
              "cond_mask" : cond_mask.permute(0,2,1),
              "observed_tp" : observed_tp}
    return inputs



# Side informations 
def time_embedding_sin(pos, d_model = 128):
    """Encode la position temporelle à chaque instant p. On calcule sin(p*omega_i) et cos(p*omega_i). 
    A chaque instant donne un vecteur de taille d_model. 
    Pour les premières fréquences, donc haute, il y a beaucoup d'oscillations donc va permettre de distinguer
    deux instants consécutifs. Pour les fréquences plus faibles, les faibles oscillations ne le permettent pas donc 
    vont permettre de donner une position à grande échelle. 
    Pour mieux comprendre on peut faire une analogie avec l'horloge, l'aiguille des secondes à une forte fréquence et celle 
    des heures, faible fréquence. 

    Args:
        pos (torch.Tensor): _description_
        d_model (int, optional): _description_. Defaults to 128.
    """
    
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)#initialisation d'un tensor de taille (B,L,d_model) avec même device que pos
    position = pos.unsqueeze(2) #(B,L,1)
    div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model) #facteurs d'échelle de fréquence sinusoïdale (omega_i)
    pe[:, :, 0::2] = torch.sin(position * div_term) #sin sur dim paire : sin(p*omega_i)
    pe[:, :, 1::2] = torch.cos(position * div_term) #cos sur dim impaire  cos(p*omega_i)
    return pe


def get_side_info(observed_tp, cond_mask,embed_layer, d_time_embedding, is_unconditional = False):
    """Permet de donner des informations au débruiteur car va recevoir en entrée des cibles très bruitées
        - attention temporel : sait où se situe chaque point sur l'axe du temps
        - attention spatial : sait quelles variables interargir entre elles
        - masque conditionnel : sait où sont les valeurs observées

    Args:
        observed_tp (_type_): _description_
        cond_mask (_type_): _description_
        embed_layer
        d_time_embedding
        is_unconditional
    """
    B, K, L = cond_mask.shape
    device = observed_tp.device
    time_embed = time_embedding_sin(observed_tp, d_time_embedding)  # (B,L,d_time_embedding)
    time_embed = time_embed.to(device)
    time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) #duplique pour chaque variables (B,L,K,d_time_embedding)
    feature_embed = embed_layer(torch.arange(K, device = cond_mask.device))  # (K,emb) une ligne par variable. identité de chaque variable
    #si deux variables ont un vecteur proche c'est qu'elles se comportent de manière similaire.
    feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) #duplique, (B,L,K,d_feature_embedding)

    side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb+d_feature_embedding)
    side_info = side_info.permute(0, 3, 2, 1)  # (Batch,channel,features,Lenght)

    if not is_unconditional:
        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1) #ajoute le masque conditionnel 

    return side_info # (B,channel,K,L)




#Construction modèle :

def conv1d(i, o, zero=False):
    """Produit de convolution noyau de taille 1. Permet d'augmenter le nombre de canaux o pour chaque case. 

    Args:
        i (_type_): _description_
        o (_type_): _description_
        zero (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    c = nn.Conv1d(i, o, 1) #produit de convolution dont le noyau est de taille 1, ne regarde une seule position à la fois
    if zero:
        nn.init.zeros_(c.weight) #tous les poids valent 0
    else:
        nn.init.kaiming_normal_(c.weight)
    return c


def make_transformer(C, n_heads):
    """Mécanisme d'attention, étudie les corrélation entre les variables 

    Args:
        C (int): Nombre de cannaux 
        n_heads (int): Nombre de heads pour paralléliser sur les cannaux

    Returns:
        _type_: _description_
    """
    layer = nn.TransformerEncoderLayer(d_model=C, nhead=n_heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(layer, num_layers=1)       # attend (S, N, C) 
    #S : longueur de séquence (axe sur lequel les éléments s'observent mutuellement S = L pour attention temporelle et S = K pour attention entre variables )


def build_model(cfg, device):
    """_summary_

    Args:
        cfg (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    C, T, d = cfg["n_channels"], cfg["n_diffusion_steps"], cfg["d_diffusion_embedding"]
    side_dim = cfg["d_time_embedding"] + cfg["d_feature_embedding"] + (0 if cfg["is_unconditional"] else 1)
    
    # C : nombre de cannaux
    # T : nombre de pas de bruit de la diffusion
    # d : taille du vecteur qui représente le pas t
    # side_dim : nombre de cannaux dans side info (temps+variables+1masque)

    # planning de bruit
    beta = np.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, T) ** 2 #bruit à ajouter sur x, progressif 
    alpha_hat = 1 - beta #part du signal conservé à l'instant t
    alpha = np.cumprod(alpha_hat) #part du signal restant après t pas
    f = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    
    # embedding sinusoïdal du pas de diffusion t : table (T, d) permet de donner le bruit actuel
    steps = torch.arange(T).unsqueeze(1).float()
    freqs = 10.0 ** (torch.arange(d // 2) / (d // 2 - 1) * 4.0).unsqueeze(0)
    table = torch.cat([torch.sin(steps * freqs), torch.cos(steps * freqs)], dim=1)
    sched = {"beta": f(beta), "alpha_hat": f(alpha_hat), "alpha": f(alpha),
             "alpha_torch": f(alpha).view(T, 1, 1), "diff_table": table.to(device)}

    net = nn.ModuleDict({
        "embed_feat": nn.Embedding(cfg["n_features"], cfg["d_feature_embedding"]),
        "diff_proj": nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()),
        "in_proj": conv1d(2, C),
        "res_layers": nn.ModuleList([nn.ModuleDict({
            "diff_proj": nn.Linear(d, C),
            "cond_proj": conv1d(side_dim, 2 * C),
            "mid_proj": conv1d(C, 2 * C),
            "out_proj": conv1d(C, 2 * C),
            "time_tf": make_transformer(C, cfg["n_heads"]),
            "feat_tf": make_transformer(C, cfg["n_heads"]),
        }) for _ in range(cfg["n_layers"])]),
        "out1": conv1d(C, C),
        "out2": conv1d(C, 1, zero=True),
    }).to(device)
    return {"net": net, "sched": sched, "cfg": cfg}


#Débruiteur 

def tf_over(y, tf, axis):
    """applique transformer selon l'axe temps ou variable. 

    Args:
        y (_type_): _description_
        tf (_type_): _description_
        axis (_type_): _description_

    Returns:
        _type_: _description_
    """
    B, C, K, L = y.shape
    if axis == "time":      # séquences de longueur L, une par (b,k)
        y = y.permute(0, 2, 1, 3).reshape(B * K, C, L)
    else:                   # séquences de longueur K, une par (b,l)
        y = y.permute(0, 3, 1, 2).reshape(B * L, C, K)
    y = tf(y.permute(2, 0, 1)).permute(1, 2, 0)
    if axis == "time":
        return y.reshape(B, K, C, L).permute(0, 2, 1, 3)
    return y.reshape(B, L, C, K).permute(0, 2, 3, 1)


def denoise(model, x, side_info, t):
    """x (B,2,K,L), side_info (B,side,K,L), t (B,) -> bruit prédit (B,K,L)*
    essaie de prédire le bruit ajouté aux données lors du processus de diffusion. 
    
    """
    net, sched = model["net"], model["sched"]
    B, _, K, L = x.shape
    x = F.relu(net["in_proj"](x.reshape(B, 2, K * L)))
    C = x.shape[1]
    x = x.reshape(B, C, K, L)
    emb = net["diff_proj"](sched["diff_table"][t])                       # (B,d)
    skips = []
    for layer in net["res_layers"]:
        y = x + layer["diff_proj"](emb).unsqueeze(-1).unsqueeze(-1)      # injecte t, informe sur le niveau de bruit
        y = tf_over(y, layer["time_tf"], "time") #capture dépendances temporelles 
        y = tf_over(y, layer["feat_tf"], "feature") #capture dépendances entre variables
        y = layer["mid_proj"](y.reshape(B, C, K * L))
        y = y + layer["cond_proj"](side_info.reshape(B, -1, K * L))      # injecte side info
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        res, skip = layer["out_proj"](y).chunk(2, dim=1)
        x = (x + res.reshape(B, C, K, L)) / math.sqrt(2)
        skips.append(skip)
    out = torch.stack(skips).sum(0) / math.sqrt(len(skips))
    out = net["out2"](F.relu(net["out1"](out)))
    return out.reshape(B, K, L)




#Loss 
def calc_loss(model, observed_data, cond_mask, target_mask, side_info, t=None):
    """Calcule le loss entre le vrai bruit et le bruit prédit par le modèle (via denoise())

    Args:
        model (_type_): _description_
        observed_data (_type_): _description_
        cond_mask (_type_): _description_
        target_mask (_type_): _description_
        side_info (_type_): _description_
        t (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    T = model["cfg"]["n_diffusion_steps"] #pas de bruit de diffusion
    B = observed_data.shape[0]
    dev = observed_data.device
    t = torch.randint(0, T, (B,), device=dev) if t is None else torch.full((B,), t, device=dev) #on prend au hasard un pas t sur [0:T]
    a = model["sched"]["alpha_torch"][t]                                 # (B,1,1) #intensité du signal par rapport au bruit
    noise = torch.randn_like(observed_data) #vérité terrain, bruit selon les vraies données 
    noisy = a.sqrt() * observed_data + (1 - a).sqrt() * noise            # diffusion avant, on crée version bruitée des données observées
    x = torch.stack([cond_mask * observed_data, (1 - cond_mask) * noisy], dim=1) #on place les données bruités là où on n'a pas de valeur 0
    pred = denoise(model, x, side_info, t) #prédiction du bruit ajoutée vi le réseau
    residual = (noise - pred) * target_mask
    return residual.pow(2).sum() / max(target_mask.sum().item(), 1) #loss entre le bruit prédit et le vrai bruit 


def calc_loss_valid(model, observed_data, cond_mask, target_mask, side_info): #Je pense faux 
    """Calcule loss pour chaque t

    Args:
        model (_type_): _description_
        observed_data (_type_): _description_
        cond_mask (_type_): _description_
        target_mask (_type_): _description_
        side_info (_type_): _description_

    Returns:
        _type_: _description_
    """
    T = model["cfg"]["n_diffusion_steps"]
    return sum(calc_loss(model, observed_data, cond_mask, target_mask, side_info, t=t)
               for t in range(T)) / T
    
    

#Diffusion inverse
@torch.no_grad()
def sample(model, observed_data, cond_mask, side_info, n_sampling_times):
    """_summary_

    Args:
        model (_type_): _description_
        observed_data (_type_): _description_
        cond_mask (_type_): _description_
        side_info (_type_): _description_
        n_sampling_times (_type_): nombre de trajectoire aléatoires que l'on veut générer, c'est-à-dire nombre d'imputations plausibles

    Returns:
        _type_: ensemble des scénarios pausibles
    """
    s, T = model["sched"], model["cfg"]["n_diffusion_steps"]
    B, K, L = observed_data.shape
    dev = observed_data.device
    out = torch.zeros(B, n_sampling_times, K, L, device=dev)
    for i in range(n_sampling_times):
        cur = torch.randn_like(observed_data) #création du bruit pur
        for t in range(T - 1, -1, -1): #on débruite de T à 0
            x = torch.stack([cond_mask * observed_data, (1 - cond_mask) * cur], dim=1)
            pred = denoise(model, x, side_info, torch.full((B,), t, device=dev))
            #on retire ce bruit du vecteur
            c1 = 1 / s["alpha_hat"][t].sqrt()
            c2 = (1 - s["alpha_hat"][t]) / (1 - s["alpha"][t]).sqrt()
            cur = c1 * (cur - c2 * pred)
            if t > 0:
                sigma = ((1 - s["alpha"][t - 1]) / (1 - s["alpha"][t]) * s["beta"][t]).sqrt()
                cur = cur + sigma * torch.randn_like(cur)
        out[:, i] = cur
    return out


#Forward (imputation)
#forward 

def forward(model, inputs, calc_criterion, training=True, n_sampling_times=1):
    cfg, results = model["cfg"], {}
    embed = model["net"]["embed_feat"]
    if calc_criterion:
        side = get_side_info(inputs["observed_tp"], inputs["cond_mask"], embed,
                             cfg["d_time_embedding"], cfg["is_unconditional"])
        fn = calc_loss if training else calc_loss_valid
        results["loss" if training else "metric"] = fn(
            model, inputs["X_ori"], inputs["cond_mask"], inputs["indicating_mask"], side)
    else:
        obs, mask = inputs["X"], inputs["cond_mask"]
        side = get_side_info(inputs["observed_tp"], mask, embed,
                             cfg["d_time_embedding"], cfg["is_unconditional"])
        samples = sample(model, obs, mask, side, n_sampling_times)      # (B,n,K,L)
        rep_obs = obs.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
        rep_mask = mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
        results["imputation"] = (rep_obs + samples * (1 - rep_mask)).permute(0, 1, 3, 2)
        results["reconstruction"] = samples.permute(0, 1, 3, 2)
    return results



def make_cfg(L, K, **cfg_kwargs):
    """Configuration par défaut du modèle"""
    cfg = dict(n_steps=L, n_features=K, n_layers=4, n_channels=64, n_heads=8,
               d_time_embedding=128, d_feature_embedding=16, d_diffusion_embedding=128,
               n_diffusion_steps=50, beta_start=1e-4, beta_end=0.5, is_unconditional=False)
    cfg.update(cfg_kwargs)
    return cfg


def masked_stats(X, O):
    """Moyenne et écart-type par variable, calculés uniquement sur les points observés.
    Args: 
         -X (T,K) valeurs (NaN ou valeurs de remplissage possibles aux points manquants),
         -O (T,K) observed_mask
    """
    Xz = np.nan_to_num(X).astype(np.float64)
    n = O.sum(0).clip(min=1)
    mean = (Xz * O).sum(0) / n
    std = np.sqrt((((Xz - mean) ** 2) * O).sum(0) / n) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X, O, mean, std):
    """Standardise avec les stats du train. Les points non observés valent 0 (= la moyenne)."""
    return (np.nan_to_num((X - mean) / std) * O).astype(np.float32)


def make_windows(arr, L, stride=None):
    """(T,K) -> (N,L,K) fenêtres glissantes."""
    stride = stride or L
    starts = range(0, arr.shape[0] - L + 1, stride)
    return np.stack([arr[s:s + L] for s in starts])


def select_windows(keep, n_max=None, seed=0):
    """Indices des fenêtres retenues (keep : tableau booléen), éventuellement sous-échantillonnés à n_max."""
    idx = np.flatnonzero(keep)
    if n_max is not None and len(idx) > n_max:
        idx = np.sort(np.random.RandomState(seed).choice(idx, n_max, replace=False))
    return idx


def fit_arrays(Xtr, Otr, Xva, Ova, Ctr=None, Cva=None, epochs=20, batch_size=16,
               lr=1e-3, patience=10, save_path="csdi_best.pt", device=None, norm=None, **cfg_kwargs):
    """Entraînement à partir de tableaux (N,L,K) déjà normalisés.
    Ctr=None : masquage aléatoire à chaque batch (get_rand_mask). Sinon cond_mask fixe fourni.
    Cva=None : masques de validation tirés une seule fois (fixes). Retourne (model, history).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, L, K = Xtr.shape
    # vérifications : un problème de données doit s'afficher clairement, pas se manifester par un early stopping silencieux
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError(f"aucune fenêtre d'entraînement ({len(Xtr)}) ou de validation ({len(Xva)}) : vérifier MIN_OBS, L et le carving")
    for nom, a in (("Xtr", Xtr), ("Xva", Xva)):
        if not np.isfinite(a).all():
            raise ValueError(f"{nom} contient des NaN ou des inf : vérifier la normalisation (variable constante, valeurs de remplissage)")
    cfg = make_cfg(L, K, **cfg_kwargs)
    model = build_model(cfg, device)
    T_ = lambda a: torch.tensor(a, dtype=torch.float32)
    tp = lambda B: torch.arange(L, device=device).float().unsqueeze(0).expand(B, L)

    if Cva is None:                                   
        torch.manual_seed(0); np.random.seed(0)
        Cva = get_rand_mask(T_(Ova)).numpy()
    if Ctr is not None and (np.asarray(Otr) - np.asarray(Ctr)).sum() <= 0:
        raise ValueError("aucune cible dans les fenêtres d'entraînement (Otr - Ctr = 0) : le carving n'y cache aucun point")
    if (np.asarray(Ova) - np.asarray(Cva)).sum() <= 0:
        raise ValueError("aucune cible dans les fenêtres de validation (Ova - Cva = 0) : la loss de validation serait toujours nulle")
    tr_tensors = [T_(Xtr), T_(Otr)] + ([T_(Ctr)] if Ctr is not None else [])
    train_loader = DataLoader(TensorDataset(*tr_tensors), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(T_(Xva), T_(Ova), T_(Cva)), batch_size=batch_size)

    opt = torch.optim.Adam(model["net"].parameters(), lr=lr, weight_decay=1e-6) #on choisi Adam optimizer
    sched_lr = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[max(1, int(0.75 * epochs)), max(1, int(0.9 * epochs))], gamma=0.1)
    best, best_state, wait, history = float("inf"), None, 0, []

    for epoch in range(epochs):
        model["net"].train()
        tr = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            X, O = batch[0].to(device), batch[1].to(device)
            cond = batch[2].to(device) if Ctr is not None else get_rand_mask(O)
            inputs = assemble_input_for_training(X, O - cond, cond, tp(X.shape[0]))
            loss = forward(model, inputs, calc_criterion=True, training=True)["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()
        sched_lr.step()
        tr /= len(train_loader)

        model["net"].eval()
        v = 0
        with torch.no_grad():
            for X, O, Cm in val_loader:
                X, O, Cm = X.to(device), O.to(device), Cm.to(device)
                inputs = assemble_input_for_training(X, O - Cm, Cm, tp(X.shape[0]))
                v += forward(model, inputs, True, training=False)["metric"].item()
        v /= len(val_loader)
        history.append({"epoch": epoch + 1, "train": tr, "val": v})
        print(f"epoch {epoch+1}/{epochs}  train={tr:.4f}  val={v:.4f}")

        if not (np.isfinite(tr) and np.isfinite(v)):    # une loss NaN/inf ne s'améliore jamais : on s'arrête avec un message clair
            raise RuntimeError(f"loss non finie à l'epoch {epoch+1} (train={tr}, val={v}) : vérifier les données (NaN, valeurs aberrantes) ou baisser lr")

        if v < best:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in model["net"].state_dict().items()}
            # pour recharger : torch.load(save_path, weights_only=False) (norm contient des arrays numpy)
            torch.save({"net": best_state, "cfg": cfg, "norm": norm}, save_path)   # enregistré dès que la validation s'améliore : rien n'est perdu si l'entraînement s'arrête
        else:
            wait += 1
            if wait >= patience:
                print("early stopping"); break

    model["net"].load_state_dict(best_state)       # best_state existe toujours ici : les loss non finies ont été interceptées plus haut
    return model, history


@torch.no_grad()
def impute(model, X, cond, n_sampling_times=5, batch_size=8):
    """Imputation avec un cond_mask explicite. X, cond : (N,L,K) normalisés. Retourne la médiane des tirages (N,L,K)."""
    if len(X) == 0:
        raise ValueError("aucune fenêtre à imputer (X est vide) : vérifier la sélection des fenêtres de test (select_windows)")
    dev = next(model["net"].parameters()).device
    model["net"].eval()
    Xin = torch.tensor(X * cond, dtype=torch.float32)       # CACHE les points carvés (sinon la vraie valeur s'ajoute à l'échantillon dans forward)
    Cm = torch.tensor(cond, dtype=torch.float32)
    outs = []
    for i in tqdm(range(0, len(Xin), batch_size), desc="imputation"):
        x, m = Xin[i:i + batch_size].to(dev), Cm[i:i + batch_size].to(dev)
        B, L, _ = x.shape
        tp = torch.arange(L, device=dev).float().unsqueeze(0).expand(B, L)
        res = forward(model, assemble_input_for_testing(x, m, tp), False, n_sampling_times=n_sampling_times)
        outs.append(res["imputation"].median(dim=1).values.cpu())    # (B,L,K)
    return torch.cat(outs).numpy()


def masked_metrics(imp, X_true, target):
    """MAE et RMSE calculés uniquement sur les cibles (target = 1)."""
    err = (imp - X_true) * target
    n = max(float(target.sum()), 1.0)
    return float(np.abs(err).sum() / n), float(np.sqrt((err ** 2).sum() / n))


def per_variable_metrics(imp, X_true, target, std, names):
    """RMSE par variable, en unités normalisées et en unités physiques."""
    n = target.sum(axis=(0, 1))
    rmse = np.sqrt((((imp - X_true) * target) ** 2).sum(axis=(0, 1)) / np.clip(n, 1, None))
    return pd.DataFrame({"n_cibles": n.astype(int), "RMSE_norm": rmse, "RMSE_phys": rmse * std}, index=names)


def interp_baseline(X, cond):
    """Baseline : interpolation linéaire par fenêtre et par variable à partir des points de cond (0 si aucun point)."""
    N, L, K = X.shape
    out = np.zeros_like(X)
    t = np.arange(L)
    for n in range(N):
        for k in range(K):
            m = cond[n, :, k] > 0
            if m.any():
                out[n, :, k] = np.interp(t, t[m], X[n, m, k])
    return out

#Plot 
def windows_to_full(arr_w, i_win, L, T_tot):
    """Aligne les fenêtres (N,L,K) dans la série complète (T_tot,K).
    Args : 
        arr_w (N,L,K) : données à aligner
        i_win : indices des fenêtres
        L (int) : nombre d'heures par fenêtre
        T_tot (int) : nombre de pas temporel
    Return :
        (T_tot, K) : série temporelle
    """
    K = arr_w.shape[-1]
    full = np.full((T_tot, K), np.nan, dtype=np.float32)
    for n, w in enumerate(i_win):
        seg = full[w * L:(w + 1) * L]            # plus court que L si la dernière fenêtre dépasse la fin de la série (lignes de remplissage)
        seg[:] = arr_w[n][:len(seg)]
    return full

def build_results(imp, lin, i_win, L, mean, std, df_test, carving_test):
    """Rassemble tout ce qu'il faut pour les graphiques, en séries complètes (T_tot,K) et en unités physiques.
    Args :
        imp (N,L,K) imputations normalisées avec CSDI
        lin (N,L,K),imptation normalisées avec interpolation linéaire
        mean (K,) : moyenne données normalisées train
        std (K,) : covariance données normalisées train 
        df_test : (DataFrame) données de test
        carving_test : (DataFrame) carving (+1 mesuré, -1 non observé, -2 caché)
    Returns :
        Dict : données prêtes pour être ploté (plot_compare_variable)
    """
    carv = carving_test.to_numpy() if hasattr(carving_test, "to_numpy") else np.asarray(carving_test)
    T_tot, K = carv.shape
    X = df_test.to_numpy(np.float32)
    true_full = np.where(carv != -1, X, np.nan)                     
    csdi_full = windows_to_full(imp * std + mean, i_win, L, T_tot)
    lin_full = windows_to_full(lin * std + mean, i_win, L, T_tot)
    mean_full = np.where(np.isnan(csdi_full), np.nan, mean).astype(np.float32) 
    hidden = carv == -2  #points cachés
    evaluated = hidden & ~np.isnan(csdi_full)                     
    print(f"points cachés : {hidden.sum()} | imputés (évalués) : {evaluated.sum()} ({evaluated.sum() / max(hidden.sum(), 1):.1%})")
    return dict(names=list(df_test.columns), dates=pd.DatetimeIndex(df_test.index), carv=carv, std=np.asarray(std),
                true=true_full, csdi=csdi_full, lin=lin_full, mean=mean_full, hidden=hidden, evaluated=evaluated)


    
def plot_compare_variable(res, var, start=None, n_hours=24 * 14):
    """
    Affiche les différentes imputations selon plusieurs méthodes, csdi, moyenne et interpolation linéaire.
    Affiche la comparaison entre les données qui étaient visibles (contexte), celles cachées et celles imputées.
    
    Args :
        var : nom de la variable
        start : date de début (ex. '2000-03-10'), sinon la période de n_hours heures qui contient le plus de points cachés."""
        
    names, dates = res["names"], res["dates"]
    if var not in names:
        raise ValueError(f"variable inconnue : {var}. Choix possibles : {names}")
    k = names.index(var)
    hid = res["evaluated"][:, k]
    if hid.sum() == 0:
        print(f"{var} : aucun point caché imputé (variable non carvée ou fenêtres non imputées)."); return
    if start is None:
        s = int(np.convolve(hid.astype(int), np.ones(n_hours, dtype=int), "valid").argmax())
    else:
        s = int(dates.get_indexer([pd.Timestamp(start)], method="nearest")[0])
    s = max(0, min(s, len(dates) - n_hours))                  
    sl = slice(s, s + n_hours)
    d, h = dates[sl], hid[sl]
    if h.sum() == 0:
        print("aucun point caché imputé dans cette période : essaie une autre date, ou start=None pour la période la plus chargée.")
        return
    true = res["true"][sl, k]
    ctx = res["carv"][sl, k] == 1

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.scatter(d[ctx], true[ctx], s=6, color="red", label="contexte visible (mesuré)", zorder=1)
    ax.plot(d, np.where(h, true, np.nan), color="tab:green", marker="o", ms=3, lw=1.3, label="valeurs cachées (vérité)", zorder=5)
    for label, key, color, marker, ls in [("CSDI", "csdi", "tab:orange", "x", "-"),
                                          ("interpolation linéaire", "lin", "tab:blue", "^", "-"),
                                          ("moyenne", "mean", "tab:red", None, "--")]:
        y = res[key][sl, k]
        rmse = np.sqrt(np.nanmean((y[h] - true[h]) ** 2)) if h.sum() else np.nan
        ax.plot(d, np.where(h, y, np.nan), color=color, marker=marker, ms=4, lw=1.1, ls=ls,
                label=f"{label} (RMSE = {rmse:.3g})", zorder=4)
    ax.set_title(f"{var} : imputation", fontweight="bold", pad=52)
    ax.set_ylabel(var); ax.grid(alpha=0.3)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); plt.show()


# ======================================================================================
# Vérification avant entraînement
# ======================================================================================
def sanity_check(Xw, Ow, device=None, **cfg_kwargs):
    """Vérifie les formes et la loss initiale sur un mini-batch (la loss doit être proche de 1 avant entraînement).
    Le masque est aléatoire ici : c'est un contrôle des formes, pas un entraînement."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, L, K = Xw.shape
    model = build_model(make_cfg(L, K, **cfg_kwargs), device)
    X, O = torch.tensor(Xw[:4]).to(device), torch.tensor(Ow[:4]).to(device)
    cond = get_rand_mask(O)
    tp = torch.arange(L, device=device).float().unsqueeze(0).expand(X.shape[0], L)
    inputs = assemble_input_for_training(X, O - cond, cond, tp)
    cfg = model["cfg"]
    side = get_side_info(inputs["observed_tp"], inputs["cond_mask"], model["net"]["embed_feat"],
                         cfg["d_time_embedding"], cfg["is_unconditional"])
    print("X_ori       :", tuple(inputs["X_ori"].shape), " (attendu : B,K,L)")
    print("side_info   :", tuple(side.shape), f" (attendu : B,{cfg['d_time_embedding'] + cfg['d_feature_embedding'] + 1},K,L)")
    loss = forward(model, inputs, True, training=True)["loss"]
    print("loss initiale :", round(loss.item(), 4), " (attendu : proche de 1)")
    print("paramètres  :", f"{sum(p.numel() for p in model['net'].parameters()):,}")


# ======================================================================================
# Test SANS carving : on impute les vrais trous (aucune valeur cachée, donc aucune erreur calculable)
# ======================================================================================
def build_imputation(imp, lin, i_win, L, mean, std, df_test, observed):
    """Remet les imputations bout à bout en séries complètes (T_tot,K), en unités physiques.

    Args :
        imp (N,L,K) : imputations normalisées de CSDI
        lin (N,L,K) ou None : interpolation linéaire normalisée (référence visuelle)
        i_win : indices des fenêtres imputées (la fenêtre n couvre les heures i_win[n]*L ... (i_win[n]+1)*L)
        mean, std (K,) : statistiques du train
        df_test (DataFrame) : données de test, dates en index
        observed (T_tot,K) : masque des points mesurés (1 = mesuré, 0 = manquant), sans lignes de remplissage
    Return :
        dict utilisé par summarize_imputation, plot_imputed_variable et plot_heatmap_imputed.
        res["final"] : série complétée (mesures conservées, trous remplis par CSDI ; NaN si la fenêtre n'a pas été imputée)
    """
    X = df_test.to_numpy(np.float32)
    T_tot, K = X.shape
    obs = np.asarray(observed)[:T_tot] > 0                                   # True = mesuré
    true_full = np.where(obs, X, np.nan)                                     # mesures (NaN aux trous)
    csdi_full = windows_to_full(imp * std + mean, i_win, L, T_tot)           # NaN hors des fenêtres imputées
    lin_full = windows_to_full(lin * std + mean, i_win, L, T_tot) if lin is not None else None
    missing = ~obs
    imputed = missing & ~np.isnan(csdi_full)                                 # trous effectivement imputés
    final = np.where(obs, X, csdi_full).astype(np.float32)
    print(f"points manquants : {missing.sum()} | imputés par CSDI : {imputed.sum()} ({imputed.sum() / max(missing.sum(), 1):.1%})")
    return dict(names=list(df_test.columns), dates=pd.DatetimeIndex(df_test.index), mean=np.asarray(mean), std=np.asarray(std),
                obs=obs, true=true_full, csdi=csdi_full, lin=lin_full, missing=missing, imputed=imputed, final=final)


def summarize_imputation(res):
    """Tableau de contrôle par variable. Sans valeur cachée, on juge la PLAUSIBILITÉ (pas l'erreur) :
      - moyenne / écart-type des mesures et des valeurs imputées (unités physiques) : doivent être du même ordre ;
      - saut au bord des trous : écart entre la mesure voisine et la valeur imputée adjacente, divisé par la variation
        horaire typique entre deux mesures consécutives. Proche de 1 : raccord continu ; très supérieur : raccord brutal ;
      - part des mesures de test hors de ±4 écarts-types du train : mesure le décalage entre train et test."""
    names, mean, std = res["names"], res["mean"], res["std"]
    rows = []
    for k, nm in enumerate(names):
        o, im = res["obs"][:, k], res["imputed"][:, k]
        x_o, x_i = res["true"][o, k], res["csdi"][im, k]
        pair = o[1:] & o[:-1]                                                # deux mesures consécutives
        step = np.abs(np.diff(res["true"][:, k]))[pair].mean() if pair.any() else np.nan
        s = np.flatnonzero(im[1:] & o[:-1]) + 1                              # 1re valeur imputée après une mesure
        e = np.flatnonzero(im[:-1] & o[1:])                                  # dernière valeur imputée avant une mesure
        jumps = np.concatenate([np.abs(res["csdi"][s, k] - res["true"][s - 1, k]),
                                np.abs(res["true"][e + 1, k] - res["csdi"][e, k])])
        rows.append({
            "manquants (%)": 100 * (~o).mean(),
            "imputés": int(im.sum()),
            "moy. mesures": x_o.mean() if len(x_o) else np.nan,
            "moy. imputées": x_i.mean() if len(x_i) else np.nan,
            "écart-type mesures": x_o.std() if len(x_o) else np.nan,
            "écart-type imputées": x_i.std() if len(x_i) else np.nan,
            "saut au bord / pas horaire": jumps.mean() / step if len(jumps) and step > 0 else np.nan,
            "mesures hors ±4σ du train (%)": 100 * (np.abs((x_o - mean[k]) / std[k]) > 4).mean() if len(x_o) else np.nan,
        })
    return pd.DataFrame(rows, index=names)


def plot_imputed_variable(res, var, start=None, n_hours=24 * 14, show_baselines=True):
    """Une variable sur une période : mesures, valeurs imputées par CSDI et (en référence visuelle) interpolation linéaire et moyenne.
    Il n'y a pas de vérité sur les trous : on regarde si l'imputation se raccorde aux mesures et garde une amplitude plausible.

    Args :
        var : nom de la variable
        start : date de début (ex. '2003-10-25'), sinon la période de n_hours heures qui contient le plus de points imputés."""
    names, dates = res["names"], res["dates"]
    if var not in names:
        raise ValueError(f"variable inconnue : {var}. Choix possibles : {names}")
    k = names.index(var)
    imp_k = res["imputed"][:, k]
    if imp_k.sum() == 0:
        print(f"{var} : aucun point imputé (aucun trou, ou fenêtres non imputées)."); return
    if start is None:
        s = int(np.convolve(imp_k.astype(int), np.ones(n_hours, dtype=int), "valid").argmax())
    else:
        s = int(dates.get_indexer([pd.Timestamp(start)], method="nearest")[0])
    s = max(0, min(s, len(dates) - n_hours))
    sl = slice(s, s + n_hours)
    d, h, o = dates[sl], imp_k[sl], res["obs"][sl, k]
    if h.sum() == 0:
        print("aucun point imputé dans cette période : essaie une autre date, ou start=None pour la période la plus chargée.")
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(d, 0, 1, where=h, transform=ax.get_xaxis_transform(), color="tab:orange", alpha=0.10, label="trou imputé")
    ax.plot(d, np.where(o, res["true"][sl, k], np.nan), color="tab:blue", marker="o", ms=2.5, lw=1, label="mesures", zorder=2)
    if show_baselines:
        if res["lin"] is not None:
            ax.plot(d, np.where(h, res["lin"][sl, k], np.nan), color="gray", marker="^", ms=3, lw=0.9, label="interpolation linéaire (référence)", zorder=3)
        ax.plot(d, np.where(h, res["mean"][k], np.nan), color="tab:red", ls="--", lw=1, label="moyenne du train (référence)", zorder=3)
    ax.plot(d, np.where(h, res["csdi"][sl, k], np.nan), color="tab:orange", marker="x", ms=4, lw=1.3, label="CSDI (valeurs imputées)", zorder=5)
    ax.set_title(f"{var} : mesures et valeurs imputées", fontweight="bold", pad=52)
    ax.set_ylabel(var); ax.grid(alpha=0.3)
    ax.set_xlabel("pas de valeur cachée : on juge si l'imputation se raccorde aux mesures et garde une amplitude plausible")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); plt.show()


def plot_heatmap_imputed(res, title_suffix=""):
    """Heatmap (même style que le masque du carving) : mesures, trous imputés par CSDI, trous non imputés."""
    col = {"mesure": "#EDE0E0", "imputé": "#88E0E0", "non_imputé": "#4B4BB0"}
    img = np.empty(res["obs"].shape + (3,)); img[:] = to_rgb(col["mesure"])
    img[res["imputed"]] = to_rgb(col["imputé"])
    img[res["missing"] & ~res["imputed"]] = to_rgb(col["non_imputé"])
    dates, names = res["dates"], res["names"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img, aspect="auto", interpolation="nearest")                   # nearest : pas de mélange de couleurs
    years = sorted(set(dates.year))
    ax.set_yticks([int(np.argmax(dates.year == y)) for y in years]); ax.set_yticklabels(years)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Variables", fontweight="bold"); ax.set_ylabel("Années", fontweight="bold")
    ax.set_title(f"Heatmap des trous imputés par CSDI{title_suffix}", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(facecolor=col["non_imputé"], edgecolor="gray", label="Manquant, non imputé"),
                       mpatches.Patch(facecolor=col["imputé"], edgecolor="gray", label="Manquant, imputé par CSDI"),
                       mpatches.Patch(facecolor=col["mesure"], edgecolor="gray", label="Mesuré")],
              loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout(); plt.show()
