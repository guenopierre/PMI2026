#%% librairies

import speasy as spz
import pandas as pd
from datetime import datetime
import numpy as np



#%%SPEASY - 1. OMNI 5 minutes dataset (high_freq)

print("be careful, it takes time")
start_time = datetime(1995, 1, 1, 0, 0)
stop_time = datetime(2026, 1, 1, 0, 0)

data_path = spz.inventories.data_tree.cda.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF__Plasma__Indices__Energetic_Proton_Flux.OMNI_HRO_5MIN
dir(data_path)

products = [data_path.BX_GSE,
            data_path.BY_GSM,
            data_path.BZ_GSM,
            data_path.Beta,
            data_path.T,
            data_path.proton_density,
            data_path.flow_speed
           ]
datasets = spz.get_data(products, start_time, stop_time)

omni_df = pd.concat([x.to_dataframe() for x in datasets], axis=1)
# omni_df.rename({'Bx, GSE':'Bx',
#                 'By, GSM':'By',
#                 'Bz, GSM':'Bz',
#                 'Plasma Beta':'Beta',
#                 'temperature':'T',
#                 'Proton density':'Np',
#                 'flow speed, GSE':'V'}, 
#                axis='columns', inplace=True)

## NaN conversion -> /!\ not really the same as the OMNI2 .dat dataset
fill_values_5min = {
    "Bx, GSE": 9999.99,
    "By, GSM": 9999.99,
    "Bz, GSM": 9999.99,
    "temperature": 9999999.0,
    "Proton density": 999.99,
    "flow speed, GSE": 99999.9,
    "Plasma Beta": 999.99
}

omni_df = omni_df.replace(fill_values_5min, np.nan)

omni_df['B'] = np.sqrt(omni_df['Bx, GSE']**2+omni_df['By, GSM']**2+omni_df['Bz, GSM']**2)

#to save
# omni_df.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/OMNI_HRO_5MIN_19952026.pkl')



#%%SPEASY - 2. OMNI 1 hour dataset (low_freq)

print("be careful, it takes a lot of time")
start_time = datetime(1995, 1, 1, 0, 0)
stop_time = datetime(2026, 1, 1, 0, 0)

# data_path = spz.inventories.data_tree.amda.Parameters.OMNI.Sun__Solar_Wind__Ground_Based_Indices.omni_hour_all
data_path = spz.inventories.data_tree.cda.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF__Plasma__Indices__Energetic_Proton_Flux.OMNI2_H0_MRG1HR
dir(data_path)


# products = [data_path.omni_imf_mag,
#             data_path.omni_imf,
#             data_path.omni_sw_beta,
#             data_path.omni_sw_t,
#             data_path.omni_sw_v,
#             data_path.omni_sw_n
#            ]
products = [data_path.BX_GSE1800,
            data_path.BY_GSM1800,
            data_path.BZ_GSM1800,
            data_path.Beta1800,
            data_path.T1800,
            data_path.N1800,
            data_path.V1800
           ]

datasets = spz.get_data(products, start_time, stop_time)

omni_df = pd.concat([x.to_dataframe() for x in datasets], axis=1)

#useless:
# omni_df.rename({'Bx, GSE':'Bx',
#                 'By, GSM':'By',
#                 'Bz, GSM':'Bz',
#                 'Plasma Beta':'Beta',
#                 'temperature':'T',
#                 'Proton density':'Np',
#                 'flow speed, GSE':'V'}, 
#                axis='columns', inplace=True)


fill_values_1h = {
    "Bx, GSE": 9999.99,
    "By, GSM": 9999.99,
    "Bz, GSM": 9999.99,
    "temperature": 9999999.0,
    "Proton density": 999.99,
    "flow speed, GSE": 99999.9,
    "Plasma Beta": 999.99
}  #I will probably have to correct it 

omni_df = omni_df.replace(fill_values_5min, np.nan)

omni_df['B'] = np.sqrt(omni_df.Bx**2+omni_df.By**2+omni_df.Bz**2)

#to save
# omni_df.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/OMNI2_H0_MRG1HR_19952026.pkl')

#%%SPEASY - 3. OMNI geomag dataset

start_time = datetime(1995, 1, 1, 0, 0)
stop_time = datetime(1996, 1, 1, 0, 0)

data_path = spz.inventories.data_tree.cda.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF__Plasma__Indices__Energetic_Proton_Flux.OMNI2_H0_MRG1HR

products = [data_path.AE1800,
            data_path.DST1800,
            data_path.KP1800,
            data_path.PC_N_INDEX1800,
            data_path.F10_INDEX1800
           ]  
datasets = spz.get_data(products, start_time, stop_time)

omni_df = pd.concat([x.to_dataframe() for x in datasets], axis=1)

omni_df.rename({'1-h AE':'AE',
                '1-h Dst':'Dst',
                '3-h Kp*10':'Kp',
                '1-h PC(N)-index':'PC_N',
                'Daily F10.7':'F_10_7'}, 
               axis='columns', inplace=True)

# omni_df.to_pickle('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_geomag1995_2026.pkl')

#%%DAT - 1. OMNI 2 Low Res (1h) - All variables & All years

omni2_low_res_all_years = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni2_all_years.dat", 
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
    "Bx_GSE,_GSM",
    "By_GSE",
    "Bz_GSE",
    "By_GSM",
    "Bz_GSM",
    "sigma|B|",
    "sigma_B",
    "sigma_Bx",
    "sigma_By",
    "sigma_Bz",
    "Proton_temperature",
    "Proton_Density",
    "Plasma_(Flow)_speed",
    "Plasma_Flow_Long._Angle",
    "Plasma_Flow_Lat._Angle",
    "Na/Np",
    "Flow_Pressure",
    "sigma_T",
    "sigma_N",
    "sigma_V",
    "sigma_phi_V",
    "sigma_theta_V",
    "sigma-Na/Np",
    "Electric_field",
    "Plasma_beta",
    "Alfven_mach_number",
    "Kp",
    "R",
    "DST_Index",
    "AE-index",
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
}
#columns and fill_values information in https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text

omni2_low_res_all_years.columns = columns
omni2_low_res_all_years = omni2_low_res_all_years.replace(fill_values, np.nan)
omni2_low_res_all_years.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 5/PMI/data/omni_1h_all.pkl")


