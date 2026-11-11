#%% Lecture des datasets

import pickle

with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_1h_all.pkl', 'rb') as file:
    # 1h (low_freq) .dat file
    omni_1h_all = pickle.load(file)
        
with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/OMNI_HRO_5MIN_19952026.pkl', 'rb') as file:
    # 5' (hig_freq) spz cda access
    OMNI_HRO_5MIN_19952026 = pickle.load(file)
    
with open('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/OMNI_H0_MRG1HR_19952026.pkl', 'rb') as file:
    # 5' (hig_freq) spz cda access
    OMNI_H0_MRG1HR_19952026 = pickle.load(file)
    
#%% Choix du dataset

df = omni_1h_all #par exemple
df = OMNI_HRO_5MIN_19952026 #par exemple
df = OMNI_H0_MRG1HR_19952026 #par exemple




#%% Conversion vers tableau PyPots

import numpy as np

def df_to_pypots_array(df):
    """PyPOTS expects X with shape (n_samples, n_steps, n_features).
    A single dataframe is treated here as ONE time-series sample."""
    return df.to_numpy(dtype=float)[np.newaxis, :, :]

X_ori = df_to_pypots_array(df)





#%% PyPOTS

import numpy as np
import pandas as pd
from tqdm import tqdm #to know where I am in the process

from pygrinder import mcar
from pypots.imputation import Mean
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre


def df_to_pypots_array(df):
    """PyPOTS expects X with shape (n_samples, n_steps, n_features).
    A single dataframe is treated here as ONE time-series sample."""
    return df.to_numpy(dtype=float)[np.newaxis, :, :]

X_ori = df_to_pypots_array(df) #to convert into the correct format for PyPOTS


for _ in tqdm(range(1), desc="Preparing data..."):
    X_ori = df_to_pypots_array(df)  # may still contain real NaNs

# -----------------------------------------------------------------
# 1. Carve out an artificial 10% extra missingness so we have
#    ground truth to actually score the imputation against.
# -----------------------------------------------------------------
for _ in tqdm(range(1), desc="Creating evaluation mask (MCAR 10%)..."):
    X = mcar(X_ori, p=0.1)
    indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # True = we hid these on purpose

test_set = {"X": X}

# -----------------------------------------------------------------
# 2. Mean imputation
# -----------------------------------------------------------------

mean_imputer = Mean()
for _ in tqdm(range(1), desc="Running Mean imputation..."):
    X_imputed = mean_imputer.impute(test_set)
    


# -----------------------------------------------------------------
# 3. Metrics (computed only on the artificially-masked values)
# -----------------------------------------------------------------
n_features = X_ori.shape[-1]
col_names = list(df.columns)
rows = []
for f in tqdm(range(n_features), desc="Scoring..."):
    pred_f = X_imputed[:, :, f:f + 1]
    true_f = np.nan_to_num(X_ori[:, :, f:f + 1])       #masks can be used for filtering. For values==0 in masks, values at their corresponding positions in predictions will be ignored.    
    mask_f = indicating_mask[:, :, f:f + 1]
    if mask_f.sum() == 0:
        continue
    rows.append({
        "feature": col_names[f],
        "MAE": calc_mae(pred_f, true_f, mask_f),
        "MSE": calc_mse(pred_f, true_f, mask_f),
        "RMSE": calc_rmse(pred_f, true_f, mask_f),
        "MRE": calc_mre(pred_f, true_f, mask_f),
    })

metrics_df = pd.DataFrame(rows).set_index("feature")
# print("\nPer-feature metrics:")
# print(metrics_df.round(4))

overall = {
    "MAE": calc_mae(X_imputed, np.nan_to_num(X_ori), indicating_mask),
    "MSE": calc_mse(X_imputed, np.nan_to_num(X_ori), indicating_mask),
    "RMSE": calc_rmse(X_imputed, np.nan_to_num(X_ori), indicating_mask),
    "MRE": calc_mre(X_imputed, np.nan_to_num(X_ori), indicating_mask),
}
print("\nOverall metrics (all features combined):")
for k, v in overall.items():
    print(f"{k}: {v:.4f}")



#%% 

import matplotlib.pyplot as plt

def find_nan_gaps(mask_1d, min_len=2):
    """Return list of (start, end) indices (end exclusive) of consecutive
    True runs in a 1D boolean array, keeping only runs >= min_len."""
    gaps = []
    start = None
    for i, val in enumerate(mask_1d):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_len:
                gaps.append((start, i))
            start = None
    if start is not None and len(mask_1d) - start >= min_len:
        gaps.append((start, len(mask_1d)))
    return gaps


def plot_imputation_gap(X_ori, X, imputation, indicating_mask, feature_idx,
                         col_name=None, min_len=3, context=15, sample=0):
    """
    Plot a gap of several consecutive artificially-hidden NaNs for one
    feature, showing original vs reconstructed (imputed) values.

    X_ori            : (n_samples, n_steps, n_features) ground truth
    X                : (n_samples, n_steps, n_features) with MCAR mask applied
    imputation       : (n_samples, n_steps, n_features) imputed output
    indicating_mask  : boolean array, True where values were artificially hidden
    feature_idx      : which feature/column to plot
    col_name         : label for the title (defaults to feature index)
    min_len          : minimum consecutive-NaN run length to look for
    context          : how many extra points to show before/after the gap
    sample           : which sample index (0 if you only have one series)
    """
    mask_1d = indicating_mask[sample, :, feature_idx]
    gaps = find_nan_gaps(mask_1d, min_len=min_len)

    if not gaps:
        raise ValueError(
            f"No artificial NaN run of length >= {min_len} found for "
            f"feature {feature_idx}. Try lowering min_len."
        )

    # take the longest gap found
    start, end = max(gaps, key=lambda g: g[1] - g[0])

    lo = max(0, start - context)
    hi = min(mask_1d.shape[0], end + context)
    t = np.arange(lo, hi)

    true_vals = X_ori[sample, lo:hi, feature_idx]
    imputed_vals = imputation[sample, lo:hi, feature_idx]
    observed_mask = ~np.isnan(X[sample, lo:hi, feature_idx])  # points NOT hidden

    label = col_name if col_name is not None else f"feature {feature_idx}"

    plt.figure(figsize=(10, 4))

    # full reconstructed curve (ground truth where observed, imputed where hidden)
    plt.plot(t, imputed_vals, color="tab:red", linewidth=1.5,
              label="Reconstructed (imputed)")
    plt.plot(t, true_vals, color="tab:blue", linewidth=1.5,
              label="Original (ground truth)")

    # mark the points that were observed vs artificially hidden
    plt.scatter(t[observed_mask], true_vals[observed_mask],
                color="tab:blue", s=25, zorder=3, label="Observed points")
    hidden_mask = ~observed_mask
    plt.scatter(t[hidden_mask], true_vals[hidden_mask],
                facecolors="none", edgecolors="black", s=60, zorder=4,
                label="Hidden (true value)")
    plt.scatter(t[hidden_mask], imputed_vals[hidden_mask],
                color="tab:red", marker="x", s=60, zorder=4,
                label="Imputed value")

    # shade the gap region
    plt.axvspan(start - 0.5, end - 0.5, color="gray", alpha=0.15,
                label=f"NaN gap ({end - start} pts)")

    plt.title(f"MCAR gap reconstruction — {label} "
              f"(gap: steps {start}-{end - 1})")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()

    return start, end


# ------------------------------------------------------------------
# Example usage with the variables from your script:
# ------------------------------------------------------------------
# Plot the largest NaN gap for the first feature (change feature_idx / col_names as needed)
feature_idx = 4
plot_imputation_gap(
    X_ori, X, X_imputed, indicating_mask,
    feature_idx=feature_idx,
    col_name=col_names[feature_idx],
    min_len=3,       # look for gaps of at least 3 consecutive NaNs
    context=15        # points to show before/after the gap
)




