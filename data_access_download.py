## FORMAT PROPOSÉ : omni_<source><version>_<resolution>_<debut>_<fin>.<data_type>

# CHAMP         VALEURS	                                 EXEMPLES
# source	    wrapper-providers, website               speasyamda, omniweb	     
# resolution	{M}min, {H}h	                         5min
# plage	        {AAAAMMJJ_AAAAMMJJ} 	                 19950101_20001231


#%% librairies

import speasy as spz
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm

#%%SPEASY
#toutes les infos dans le Excel 'datasets [...]' sur le Teams (paths, names, missing_values, etc.)

def speasy_access(
    path,
    products,  
    start_time=datetime(1963, 1, 1, 0, 0),
    stop_time=datetime(2025, 12, 31, 23, 59),
    fill_nan=True,
):
  # Résolution des produits
  if isinstance(products, str):
    products = [products]

  resolved_products = []
  for p in products:
    if isinstance(p, str):
      resolved_products.append(getattr(path, p))
    else:
      resolved_products.append(p)

  # Téléchargement avec tqdm
  datasets = []
  for prod in tqdm(
      resolved_products, desc="Téléchargement des produits Speasy"
  ):
    datasets.append(spz.get_data(prod, start_time, stop_time))

  # Concaténation propre (les produits partagent la même grille temporelle)
  df = pd.concat([x.to_dataframe() for x in datasets], axis=1)
  
  if fill_nan:
    fill_values = {
        # High freq (5')
        "Bx, GSE": 9999.99,
        "By, GSM": 9999.99,
        "Bz, GSM": 9999.99,
        "Plasma Beta": 999.99,
        "Na/Np ratio": 9.999,
        "Flow pressure": 99.99,
        "flow speed, GSE": 9999.9,
        "Proton density": 999.99,
        "temperature": 9.99999e+06,
        "5-m AE": 99999,
        
        # Low freq (1H)
        "1AU IP Bx, GSE": 999.9,
        "1AU IP By, GSM": 999.9,
        "1AU IP Bz, GSM": 999.9,
        "1AU IP Plasma Beta": 999.99,
        "1AU IP N (ion)": 999.9,
        "1AU IP Flow Pressure": 99.99,
        "1AU IP Alpha/Proton": 9.999,
        "1AU IP Plasma Temp": 9.999999e+06,
        "1AU IP Plasma Speed": 9999.0, 
        "1-h AE" : 9999
    }
    for col, val in tqdm(
        fill_values.items(), desc="Remplacement des valeurs NaN"
    ):
      if col in df.columns:
        df[col] = df[col].replace(val, np.nan)
        
    # Calcul B_tot
    ## high freq: 
    with tqdm(total=1, desc="Calcul du champ magnétique B") as pbar:
        if {"Bx, GSE", "By, GSM", "Bz, GSM"}.issubset(df.columns):
            df["B"] = np.sqrt(
                df["Bx, GSE"] ** 2 + df["By, GSM"] ** 2 + df["Bz, GSM"] ** 2
            )
        elif {"1AU IP Bx, GSE", "1AU IP By, GSM", "1AU IP Bz, GSM"}.issubset(df.columns):
            df["B"] = np.sqrt(
                df["1AU IP Bx, GSE"] ** 2 + df["1AU IP By, GSM"] ** 2 + df["1AU IP Bz, GSM"] ** 2
            )
        pbar.update(1)

  return datasets, df


## PATHS
path_omni_speasycda1_5min  = spz.inventories.data_tree.cda.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF__Plasma__Indices__Energetic_Proton_Flux.OMNI_HRO_5MIN
path_omni_speasycda2_5min  = spz.inventories.data_tree.cda.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF__Plasma__Indices__Energetic_Proton_Flux.OMNI_HRO2_5MIN
path_omni_speasycda2_1h  = spz.inventories.data_tree.cda.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF__Plasma__Indices__Energetic_Proton_Flux.OMNI2_H0_MRG1HR


## PRODUCTS
#accès de toute la liste des produits avec la commande dir(path_{...})
products_omni_speasycda1_5min = ["BX_GSE",
                                "BY_GSM",
                                "BZ_GSM",
                                "proton_density",
                                "Beta", 
                                "NaNp_Ratio",
                                "flow_speed",
                                "AE_INDEX"
                                ]

products_omni_speasycda2_5min = ['AE_INDEX', 
                                 'BX_GSE',
                                 'BY_GSM',
                                 'BZ_GSM',
                                 'Beta',
                                 'NaNp_Ratio',
                                 'Pressure',
                                 'flow_speed',
                                 'proton_density', 
                                 'T'
                                ]

products_omni_speasycda2_1h = ['AE1800',
                             'BX_GSE1800',
                             'BY_GSM1800',
                             'BZ_GSM1800',
                             'Beta1800',
                             'DST1800',
                             'KP1800',
                             'N1800',
                             'Pressure1800',
                             'Ratio1800',
                             'T1800',
                             'V1800'
                            ]

#Fyi, le cda 1h de 1984 à 1995 m'a pris 45' à télécharger
[_, omni_speasycda2_1h_20080101_20191231] = speasy_access(path_omni_speasycda2_1h, products_omni_speasycda2_1h, 
                                                          start_time=datetime(2008, 1, 1, 0, 0), 
                                                          stop_time=datetime(2019, 12, 31, 23, 59))
    
#to save (reste à faire de 1972 à 1983)
# omni_speasycda2_1h_20080101_20191231.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_speasycda2_1h_20080101_20191231.pkl')


#%%OMNIWEB

omni_omniweb2_5min_19630101_19920129 = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni2_all_years.dat", 
                 sep="\s+", 
                 header=None)  
#omni2_all_years.dat accessible here: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat

columns = [
    "Year",
    "Decimal_Day",
    "Hour",
    "Bartels_rotation_number",
    "ID_for_IMF_spacecraft",
    "ID_for_SW_plasma_spacecraft",
    "#_of_points_in_the_IMF_averages",
    "#_of_points_in_the_plasma_averages",
    "Field_Magnitude_Average_|B|",
    "Magnitude_of_Average_Field_Vector",
    "Lat.Angle_of_Aver._Field_Vector",
    "Long.Angle_of_Aver.Field_Vector",
    "1AU IP Bx, GSE",
    "By_GSE",
    "Bz_GSE",
    "1AU IP By, GSM",
    "1AU IP Bz, GSM",
    "sigma|B|",
    "sigma_B",
    "sigma_Bx",
    "sigma_By",
    "sigma_Bz",
    "1AU IP Plasma Temp",
    "1AU IP N (ion)",
    "1AU IP Plasma Speed",
    "Plasma_Flow_Long._Angle",
    "Plasma_Flow_Lat._Angle",
    "1AU IP Alpha/Proton",
    "1AU IP Flow Pressure",
    "sigma_T",
    "sigma_N",
    "sigma_V",
    "sigma_phi_V",
    "sigma_theta_V",
    "sigma-Na/Np",
    "Electric_field",
    "1AU IP Plasma Beta",
    "Alfven_mach_number",
    "3-h Kp*10",
    "R",
    "1-h Dst",
    "1-h AE",
    "Proton_flux_>1_Mev",
    "Proton_flux_>2_Mev",
    "Proton_flux_>4_Mev",
    "Proton_flux_>10_Mev",
    "Proton_flux_>30_Mev",
    "Proton_flux_>60_Mev",
    "Flag(***)",
    "ap-index",
    "f10.7_index",
    "PC(N)_index",
    "AL-index,_from_Kyoto",
    "AU-index,_from_Kyoto",
    "Magnetosonic_mach_number",
]

fill_values = {
    "Bartels_rotation_number": 9999,
    "ID_for_IMF_spacecraft": 99,
    "ID_for_SW_plasma_spacecraft": 99,
    "#_of_points_in_the_IMF_averages": 999,
    "#_of_points_in_the_plasma_averages": 999,
    "Field_Magnitude_Average_|B|": 999.9,
    "Magnitude_of_Average_Field_Vector": 999.9,
    "Lat.Angle_of_Aver._Field_Vector": 999.9,
    "Long.Angle_of_Aver.Field_Vector": 999.9,
    "Bx_GSE,_GSM": 999.9,
    "By_GSE": 999.9,
    "Bz_GSE": 999.9,
    "By_GSM": 999.9,
    "Bz_GSM": 999.9,
    "sigma|B|": 999.9,
    "sigma_B": 999.9,
    "sigma_Bx": 999.9,
    "sigma_By": 999.9,
    "sigma_Bz": 999.9,
    "Proton_temperature": 9999999.0,
    "Proton_Density": 999.9,
    "Plasma_(Flow)_speed": 9999.0,
    "Plasma_Flow_Long._Angle": 999.9,
    "Plasma_Flow_Lat._Angle": 999.9,
    "Na/Np": 9.999,
    "Flow_Pressure": 99.99,
    "sigma_T": 9999999.0,
    "sigma_N": 999.9,
    "sigma_V": 9999.0,
    "sigma_phi_V": 999.9,
    "sigma_theta_V": 999.9,
    "sigma-Na/Np": 9.999,
    "Electric_field": 999.99,
    "Plasma_beta": 999.99,
    "Alfven_mach_number": 999.9,
    "Kp": 99,
    "R": 999,
    "DST_Index": 99999,
    "AE-index": 9999,
    "Proton_flux_>1_Mev": 999999.99,
    "Proton_flux_>2_Mev": 99999.99,
    "Proton_flux_>4_Mev": 99999.99,
    "Proton_flux_>10_Mev": 99999.99,
    "Proton_flux_>30_Mev": 99999.99,
    "Proton_flux_>60_Mev": 99999.99,
    "Flag(***)": 0,
    "ap-index": 999,
    "f10.7_index": 999.9,
    "PC(N)_index": 999.9,
    "AL-index,_from_Kyoto": 99999,
    "AU-index,_from_Kyoto": 99999,
    "Magnetosonic_mach_number": 99.9,
    "1AU IP Bx, GSE": 999.9,
    "1AU IP By, GSM": 999.9,
    "1AU IP Bz, GSM": 999.9,
    "1AU IP Plasma Beta": 999.99,
    "1AU IP N (ion)": 999.9,
    "1AU IP Flow Pressure": 99.99,
    "1AU IP Alpha/Proton": 9.999,
    "1AU IP Plasma Temp": 9.99999e+06,
    "1AU IP Plasma Speed": 9999.0
}
#columns and fill_values information in https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text

omni_omniweb2_5min_19630101_19920129.columns = columns
omni_omniweb2_5min_19630101_19920129 = omni_omniweb2_5min_19630101_19920129.replace(fill_values, np.nan)

# to save
omni_omniweb2_5min_19630101_19920129.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_omniweb2_5min_19630101_19920129.pkl")


