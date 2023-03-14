#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#                       神兽保佑
#                      代码无BUG!

"""

    Code for gejinming's essay
    Mainly for AOD, IWP, COD data segmentation and analysis
    
    The util function is in muqy_20221224_util_Essay_plot_func.py
    
    Owner: Mu Qingyu
    version 1.0
          
    Created: 2022-12-14
    
    Including the following parts:

        1) Read in AOD, IWP, COD data
                
        2) Preprocess the data in main gap and aux gap
        
        3) Plot the line plot and spatial plot together
        
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
from cartopy.io.shapereader import Reader
from matplotlib import rcParams

from muqy_20221224_util_Essay_plot_func import *

# set the font and style
# double check the font and style
mpl.rcParams["font.family"] = "sans-serif"
mpl.style.use("seaborn-ticks")


# ----- import data -----
# Read in AER data
data_AER = read_in_data("Data/AOD_bilinear_2010-2020.nc")
# Read in CLD data
data_CLD = read_in_data("Data/CERES_cloud.nc")
# Read in PC data
data_PC = read_in_data("Data/EOF_T300_RH300_U300_w300_se300.nc")
# Read in PRE data
data_PRE = read_in_data("Data/GPM_IMGER_2010-2020.nc")
# Read in radi data
data_radi = read_in_data("Data/CERES_radi.nc")

# ----- read in AOD\IWP\COD data -----
# The code below reads the data from the netcdf files into the variables.
# The data is read from the netcdf files into the variables data_AOD, data_IWP, data_IPR, data_HCF, data_COD, data_PC, data_PRE, data_radi_net_clr, data_radi_net_all, data_radi_sw_clr, data_radi_sw_all, data_radi_lw_clr, and data_radi_lw_all.
# The data is read from the netcdf files in the following order: data_AOD, data_IWP, data_IPR, data_HCF, data_COD, data_PC, data_PRE, data_radi_net_clr, data_radi_net_all, data_radi_sw_clr, data_radi_sw_all, data_radi_lw_clr, and data_radi_lw_all.
# The data is stored in the following order: data_AOD, data_IWP, data_IPR, data_HCF, data_COD, data_PC, data_PRE, data_radi_net_clr, data_radi_net_all, data_radi_sw_clr, data_radi_sw_all, data_radi_lw_clr, and data_radi_lw_all.


def extract_data(data_AER, data_CLD, data_PC, data_PRE, data_radi):
    data_AOD = data_AER["AOD_bilinear"][:, ::-1, :]
    data_IWP = data_CLD["iwp_high_daily"]
    data_IPR = data_CLD["cldicerad_high_daily"]
    data_HCF = data_CLD["cldarea_high_daily"]
    data_COD = data_CLD["cldtau_high_daily"]
    data_PC = data_PC["EOF"][0, :, :, :]
    data_PRE = data_PRE["preicp"]
    data_radi_net_clr = data_radi["toa_net_clr_daily"][0:4018]
    data_radi_net_all = data_radi["toa_net_all_daily"][0:4018]

    data_radi_sw_clr = data_radi["toa_sw_clr_daily"][0:4018]
    data_radi_sw_all = data_radi["toa_sw_all_daily"][0:4018]

    data_radi_lw_clr = data_radi["toa_lw_clr_daily"][0:4018]
    data_radi_lw_all = data_radi["toa_lw_all_daily"][0:4018]
    return (
        data_PC,
        data_PRE,
        data_AOD,
        data_IWP,
        data_IPR,
        data_HCF,
        data_COD,
        data_radi_net_clr,
        data_radi_net_all,
        data_radi_sw_clr,
        data_radi_sw_all,
        data_radi_lw_clr,
        data_radi_lw_all,
    )


(
    data_PC,
    data_PRE,
    data_AOD,
    data_IWP,
    data_IPR,
    data_HCF,
    data_COD,
    data_radi_net_clr,
    data_radi_net_all,
    data_radi_sw_clr,
    data_radi_sw_all,
    data_radi_lw_clr,
    data_radi_lw_all,
) = extract_data(data_AER, data_CLD, data_PC, data_PRE, data_radi)

# specify the region
# Extract the data of interest
# (AOD, IWP, COD, PRE, IPR, HCF, radi_net_clr, radi_net_all, radi_sw_clr, radi_sw_all, radi_lw_clr, radi_lw_all)
# and filter the data with latitude and longitude
def select_region(
    data, latS=24.5, latN=40.5, lonL=69.5, lonR=105.5
):
    # Select region of interest in the data
    # Input: data = dataset
    #        latS = latitude of the southernmost point
    #        latN = latitude of the northernmost point
    #        lonL = longitude of the westernmost point
    #        lonR = longitude of the easternmost point
    # Output: data = dataset of the selected region
    data = data.sel(lat=slice(latS, latN), lon=slice(lonL, lonR))
    return data


data_AOD = select_region(data_AOD)
data_IWP = select_region(data_IWP)
data_COD = select_region(data_COD)
data_PRE = select_region(data_PRE)
data_IPR = select_region(data_IPR)
data_HCF = select_region(data_HCF)
data_radi_net_clr = select_region(data_radi_net_clr)
data_radi_net_all = select_region(data_radi_net_all)
data_radi_sw_clr = select_region(data_radi_sw_clr)
data_radi_sw_all = select_region(data_radi_sw_all)
data_radi_lw_clr = select_region(data_radi_lw_clr)
data_radi_lw_all = select_region(data_radi_lw_all)


# Flatten the data
def flatten_data(data):
    # Function: flatten_data
    # Purpose: Flatten the data into an array of values
    # Input: data
    # Output: flattened data
    return data.values.flatten()


data_AOD = flatten_data(data_AOD)
data_IWP = flatten_data(data_IWP)
data_COD = flatten_data(data_COD)
data_PC = flatten_data(data_PC)
data_PRE = flatten_data(data_PRE)
data_IPR = flatten_data(data_IPR)
data_HCF = flatten_data(data_HCF)
data_radi_net_clr = flatten_data(data_radi_net_clr)
data_radi_net_all = flatten_data(data_radi_net_all)
data_radi_sw_clr = flatten_data(data_radi_sw_clr)
data_radi_sw_all = flatten_data(data_radi_sw_all)
data_radi_lw_clr = flatten_data(data_radi_lw_clr)
data_radi_lw_all = flatten_data(data_radi_lw_all)

# calculate the cloud radiative forcing
data_CRF = data_radi_net_clr - data_radi_net_all
data_CRF_sw = data_radi_sw_clr - data_radi_sw_all
data_CRF_lw = data_radi_lw_clr - data_radi_lw_all

# this is critical, only filter the data with AOD < 0.06
# can reproduce the results in the paper
data_AOD[data_AOD < 0.06] = np.nan
data_PRE[data_PRE == 0] = np.nan

# for the PC data woth nan, set the symtinous
# AOD, IWP, COD data to nan
# now the data are all 1d array
def give_nan_value(
    data_PC,
    data_PRE,
    data_AOD,
    data_IWP,
    data_IPR,
    data_HCF,
    data_COD,
    data_CRF,
    data_CRF_sw,
    data_CRF_lw,
):
    data_AOD[np.isnan(data_PC)] = np.nan
    data_IWP[np.isnan(data_PC)] = np.nan
    data_COD[np.isnan(data_PC)] = np.nan
    data_PRE[np.isnan(data_PC)] = np.nan
    data_IPR[np.isnan(data_PC)] = np.nan
    data_HCF[np.isnan(data_PC)] = np.nan
    data_CRF[np.isnan(data_PC)] = np.nan
    data_CRF_sw[np.isnan(data_PC)] = np.nan
    data_CRF_lw[np.isnan(data_PC)] = np.nan

    if np.isnan(data_AOD).any():
        data_AOD[np.isnan(data_IWP)] = np.nan
        data_COD[np.isnan(data_IWP)] = np.nan
        data_AOD[np.isnan(data_COD)] = np.nan
        data_IWP[np.isnan(data_COD)] = np.nan
        data_IWP[np.isnan(data_AOD)] = np.nan
        data_COD[np.isnan(data_AOD)] = np.nan
        data_PRE[np.isnan(data_AOD)] = np.nan
        data_PRE[np.isnan(data_IWP)] = np.nan
        data_PRE[np.isnan(data_COD)] = np.nan
        data_IPR[np.isnan(data_AOD)] = np.nan
        data_IPR[np.isnan(data_IWP)] = np.nan
        data_IPR[np.isnan(data_COD)] = np.nan
        data_HCF[np.isnan(data_AOD)] = np.nan
        data_HCF[np.isnan(data_IWP)] = np.nan
        data_HCF[np.isnan(data_COD)] = np.nan
        data_CRF[np.isnan(data_AOD)] = np.nan
        data_CRF[np.isnan(data_IWP)] = np.nan
        data_CRF[np.isnan(data_COD)] = np.nan
        data_CRF_sw[np.isnan(data_AOD)] = np.nan
        data_CRF_sw[np.isnan(data_IWP)] = np.nan
        data_CRF_sw[np.isnan(data_COD)] = np.nan
        data_CRF_lw[np.isnan(data_AOD)] = np.nan
        data_CRF_lw[np.isnan(data_IWP)] = np.nan
        data_CRF_lw[np.isnan(data_COD)] = np.nan


give_nan_value(
    data_PC,
    data_PRE,
    data_AOD,
    data_IWP,
    data_IPR,
    data_HCF,
    data_COD,
    data_CRF,
    data_CRF_sw,
    data_CRF_lw,
)

####################################################
####################################################
##### DataArrayPreprocess class ####################
###################################################

# region

####### IWP\PC1 gap AOD-COD #############################
# IWP gap AOD-COD
dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_COD,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean_0,
    dataarray_COD_gap_mean_0,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_0,
    intercept_0,
    p_value_0,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_0, dataarray_COD_gap_mean_0
)
intercept_0 = np.exp(intercept_0)

# spatial split of linear regression
(
    slope_spatial_split_0,
    p_spatial_split_0,
) = dataarray_sort_split.dataarray_spatial_split()

# PC1 gap AOD-COD
dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_COD,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean_1,
    dataarray_COD_gap_mean_1,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_1,
    intercept_1,
    p_value_1,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_1, dataarray_COD_gap_mean_1
)
intercept_1 = np.exp(intercept_1)

# spatial split of linear regression
(
    slope_spatial_split_1,
    p_spatial_split_1,
) = dataarray_sort_split.dataarray_spatial_split()


plot_both_pc1_iwp(
    # figure 1
    # line plot
    data_x_0=dataarray_AOD_gap_mean_0,
    data_y_0=dataarray_COD_gap_mean_0,
    main_gap_0=IWP_gap,
    slope_0=slope_0,
    intercept_0=intercept_0,
    yticks_0=[-2, -1, 0, 1, 2],
    main_gap_name_0="IWP",
    ymin_0=-5.8,
    ymax_0=2.4,
    xmin_0=-2.9,
    xmax_0=-1.1,
    # spatial split
    dataarray_0=slope_spatial_split_0,
    p_spatial_split_0=p_spatial_split_0,
    vmin_0=-0.94,
    vmax_0=0.94,
    # figure 2
    # line plot
    data_x_1=dataarray_AOD_gap_mean_1,
    data_y_1=dataarray_COD_gap_mean_1,
    main_gap_1=PC_gap,
    slope_1=slope_1,
    intercept_1=intercept_1,
    yticks_1=[-1, 0, 1, 2],
    main_gap_name_1="PC1",
    ymin_1=-4.1,
    ymax_1=2.2,
    xmin_1=-2.9,
    xmax_1=-1.1,
    # spatial split
    dataarray_1=slope_spatial_split_1,
    p_spatial_split_1=p_spatial_split_1,
    vmin_1=-1.7,
    vmax_1=1.7,
    # Universal settings
    data_x_label="ln(AOD)",
    data_y_label="ln(COD)",
    x_var_name="AOD",
    y_var_name="COD",
    linewidth=3,
    cmap_file="Color/PC1_color.txt",
)

####### IWP\PC1 gap AOD-IPR #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_IPR,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean_0,
    dataarray_IPR_gap_mean_0,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_0,
    intercept_0,
    p_value_0,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_0, dataarray_IPR_gap_mean_0
)
intercept_0 = np.exp(intercept_0)

# spatial split of linear regression
(
    slope_spatial_split_0,
    p_spatial_split_0,
) = dataarray_sort_split.dataarray_spatial_split()

# PC1 gap AOD-IPR
dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_IPR,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean_1,
    dataarray_IPR_gap_mean_1,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_1,
    intercept_1,
    p_value_1,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_1, dataarray_IPR_gap_mean_1
)
intercept_1 = np.exp(intercept_1)

# spatial split of linear regression
(
    slope_spatial_split_1,
    p_spatial_split_1,
) = dataarray_sort_split.dataarray_spatial_split()


plot_both_pc1_iwp(
    # figure 1
    # line plot
    data_x_0=dataarray_AOD_gap_mean_0,
    data_y_0=dataarray_IPR_gap_mean_0,
    main_gap_0=IWP_gap,
    slope_0=slope_0,
    intercept_0=intercept_0,
    yticks_0=[3.25, 3.3, 3.35, 3.4, 3.45],
    main_gap_name_0="IWP",
    ymin_0=3.0,
    ymax_0=3.47,
    xmin_0=-2.9,
    xmax_0=-1.1,
    # spatial split
    dataarray_0=slope_spatial_split_0,
    p_spatial_split_0=p_spatial_split_0,
    vmin_0=-0.2,
    vmax_0=0.2,
    # figure 2
    # line plot
    data_x_1=dataarray_AOD_gap_mean_1,
    data_y_1=dataarray_IPR_gap_mean_1,
    main_gap_1=PC_gap,
    slope_1=slope_1,
    intercept_1=intercept_1,
    yticks_1=[3.34, 3.36, 3.38],
    main_gap_name_1="PC1",
    ymin_1=3.28,
    ymax_1=3.395,
    xmin_1=-2.9,
    xmax_1=-1.1,
    # spatial split
    dataarray_1=slope_spatial_split_1,
    p_spatial_split_1=p_spatial_split_1,
    vmin_1=-0.2,
    vmax_1=0.2,
    # Universal settings
    data_x_label="ln(AOD)",
    data_y_label="ln(IPR)",
    x_var_name="AOD",
    y_var_name="IPR",
    linewidth=3,
    cmap_file="Color/PC1_color.txt",
)

####### IWP gap AOD-HCF #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_HCF,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean_0,
    dataarray_HCF_gap_mean_0,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_0,
    intercept_0,
    p_value_0,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_0, dataarray_HCF_gap_mean_0
)
intercept_0 = np.exp(intercept_0)

# spatial split of linear regression
(
    slope_spatial_split_0,
    p_spatial_split_0,
) = dataarray_sort_split.dataarray_spatial_split()


# PC1 gap AOD-HCF
dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_HCF,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean_1,
    dataarray_HCF_gap_mean_1,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_1,
    intercept_1,
    p_value_1,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_1, dataarray_HCF_gap_mean_1
)
intercept_1 = np.exp(intercept_1)

# spatial split of linear regression
(
    slope_spatial_split_1,
    p_spatial_split_1,
) = dataarray_sort_split.dataarray_spatial_split()


plot_both_pc1_iwp(
    # figure 1
    # line plot
    data_x_0=dataarray_AOD_gap_mean_0,
    data_y_0=dataarray_HCF_gap_mean_0,
    main_gap_0=IWP_gap,
    slope_0=slope_0,
    intercept_0=intercept_0,
    yticks_0=[0, 1, 2, 3],
    main_gap_name_0="IWP",
    ymin_0=-2.6,
    ymax_0=3.5,
    xmin_0=-2.9,
    xmax_0=-1.1,
    # spatial split
    dataarray_0=slope_spatial_split_0,
    p_spatial_split_0=p_spatial_split_0,
    vmin_0=-2,
    vmax_0=2,
    # figure 2
    # line plot
    data_x_1=dataarray_AOD_gap_mean_1,
    data_y_1=dataarray_HCF_gap_mean_1,
    main_gap_1=PC_gap,
    slope_1=slope_1,
    intercept_1=intercept_1,
    yticks_1=[0, 1, 2, 3],
    main_gap_name_1="PC1",
    ymin_1=-2.6,
    ymax_1=3.9,
    xmin_1=-2.9,
    xmax_1=-1.1,
    # spatial split
    dataarray_1=slope_spatial_split_1,
    p_spatial_split_1=p_spatial_split_1,
    vmin_1=-2,
    vmax_1=2,
    # Universal settings
    data_x_label="ln(AOD)",
    data_y_label="ln(HCF)",
    x_var_name="AOD",
    y_var_name="HCF",
    linewidth=3,
    cmap_file="Color/PC1_color.txt",
)

####### PC1 gap AOD-IWP #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_IWP,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean,
    dataarray_IWP_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_IWP_gap_mean
)
intercept = np.exp(intercept)

# spatial split of linear regression
(
    slope_spatial_split,
    p_spatial_split,
) = dataarray_sort_split.dataarray_spatial_split()

# ---- plot --------------------
plot_both(
    data_x=dataarray_AOD_gap_mean,
    data_y=dataarray_IWP_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[2.5, 3, 3.5, 4, 4.5, 5],
    data_x_label="ln(AOD)",
    data_y_label="ln(IWP)",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="IWP",
    ymin=-0.5,
    ymax=5.5,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-2,
    max=2,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)


####### IWP gap AOD-PRE #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_PRE,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean_0,
    dataarray_PRE_gap_mean_0,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_0,
    intercept_0,
    p_value_0,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_0, dataarray_PRE_gap_mean_0
)
intercept_0 = np.exp(intercept_0)

# spatial split of linear regression
(
    slope_spatial_split_0,
    p_spatial_split_0,
) = dataarray_sort_split.dataarray_spatial_split()

####### PC1 gap AOD-PRE #############################
dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_PRE,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean_1,
    dataarray_PRE_gap_mean_1,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_1,
    intercept_1,
    p_value_1,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_1, dataarray_PRE_gap_mean_1
)
intercept_1 = np.exp(intercept_1)

# spatial split of linear regression
(
    slope_spatial_split_1,
    p_spatial_split_1,
) = dataarray_sort_split.dataarray_spatial_split()


plot_both_pc1_iwp(
    # figure 1
    # line plot
    data_x_0=dataarray_AOD_gap_mean_0,
    data_y_0=dataarray_PRE_gap_mean_0,
    main_gap_0=IWP_gap,
    slope_0=slope_0,
    intercept_0=intercept_0,
    yticks_0=[-3, -2, -1, 0, 1],
    main_gap_name_0="IWP",
    ymin_0=-7.4,
    ymax_0=1.1,
    xmin_0=-2.9,
    xmax_0=-1.1,
    # spatial split
    dataarray_0=slope_spatial_split_0,
    p_spatial_split_0=p_spatial_split_0,
    vmin_0=-3,
    vmax_0=3,
    # figure 2
    # line plot
    data_x_1=dataarray_AOD_gap_mean_1,
    data_y_1=dataarray_PRE_gap_mean_1,
    main_gap_1=PC_gap,
    slope_1=slope_1,
    intercept_1=intercept_1,
    yticks_1=[-3, -2, -1, 0, 1],
    main_gap_name_1="PC1",
    ymin_1=-7.5,
    ymax_1=1.1,
    xmin_1=-2.9,
    xmax_1=-1.1,
    # spatial split
    dataarray_1=slope_spatial_split_1,
    p_spatial_split_1=p_spatial_split_1,
    vmin_1=-3.2,
    vmax_1=3.2,
    # Universal settings
    data_x_label="ln(AOD)",
    data_y_label="ln(PRE)",
    x_var_name="AOD",
    y_var_name="PRE",
    linewidth=3,
    cmap_file="Color/PC1_color.txt",
)

####### IWP gap AOD-CRF #############################

dataarray_sort_split = DataArrayPreprocessNoLog(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_CRF,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean_0,
    dataarray_CRF_gap_mean_0,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_0,
    intercept_0,
    p_value_0,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_0, dataarray_CRF_gap_mean_0
)
# intercept = np.exp(intercept)

# spatial split of linear regression
(
    slope_spatial_split_0,
    p_spatial_split_0,
) = dataarray_sort_split.dataarray_spatial_split()

####### PC1 gap AOD-CRF #############################

dataarray_sort_split = DataArrayPreprocessNoLog(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_CRF,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean_1,
    dataarray_CRF_gap_mean_1,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope_1,
    intercept_1,
    p_value_1,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean_1, dataarray_CRF_gap_mean_1,
)

# spatial split of linear regression
(
    slope_spatial_split_1,
    p_spatial_split_1,
) = dataarray_sort_split.dataarray_spatial_split()


plot_both_nolog_pc1_iwp(
    # figure 1
    # line plot
    data_x_0=dataarray_AOD_gap_mean_0,
    data_y_0=dataarray_CRF_gap_mean_0,
    main_gap_0=IWP_gap,
    slope_0=slope_0,
    intercept_0=intercept_0,
    yticks_0=[0, 20, 40, 60,],
    main_gap_name_0="IWP",
    ymin_0=-65,
    ymax_0=72,
    xmin_0=0.03,
    xmax_0=0.32,
    # spatial split
    dataarray_0=slope_spatial_split_0,
    p_spatial_split_0=p_spatial_split_0,
    vmin_0=-400,
    vmax_0=400,
    # figure 2
    # line plot
    data_x_1=dataarray_AOD_gap_mean_1,
    data_y_1=dataarray_CRF_gap_mean_1,
    main_gap_1=PC_gap,
    slope_1=slope_1,
    intercept_1=intercept_1,
    yticks_1=[0, 20, 40,],
    main_gap_name_1="PC1",
    ymin_1=-10,
    ymax_1=47,
    xmin_1=0.03,
    xmax_1=0.32,
    # spatial split
    dataarray_1=slope_spatial_split_1,
    p_spatial_split_1=p_spatial_split_1,
    vmin_1=-400,
    vmax_1=400,
    # Universal settings
    data_x_label="AOD",
    data_y_label="CRF",
    x_var_name="AOD",
    y_var_name="CRF",
    linewidth=3,
    cmap_file="Color/PC1_color.txt",
)

####### IWP gap AOD-CRF_sw #############################

dataarray_sort_split = DataArrayPreprocessNoLog(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_CRF_sw,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean,
    dataarray_CRF_sw_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_CRF_sw_gap_mean
)
# intercept = np.exp(intercept)

# spatial split of linear regression
(
    slope_spatial_split,
    p_spatial_split,
) = dataarray_sort_split.dataarray_spatial_split()

# ---- plot --------------------
plot_both_nolog_swlw(
    data_x=dataarray_AOD_gap_mean,
    data_y=dataarray_CRF_sw_gap_mean,
    main_gap=IWP_gap,
    slope=slope,
    intercept=intercept,
    yticks=[-110, -70, -30],
    data_x_label="AOD",
    data_y_label="SWCRF",
    main_gap_name="IWP",
    x_var_name="AOD",
    y_var_name="SWCRF",
    ymin=-190,
    ymax=-20,
    xmin=0.03,
    xmax=0.32,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-400,
    max=400,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)


####### PC1 gap AOD-CRF_sw #############################

dataarray_sort_split = DataArrayPreprocessNoLog(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_CRF_sw,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean,
    dataarray_CRF_sw_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_CRF_sw_gap_mean,
)

# spatial split of linear regression
(
    slope_spatial_split,
    p_spatial_split,
) = dataarray_sort_split.dataarray_spatial_split()

# ---- plot --------------------
plot_both_nolog_swlw(
    data_x=dataarray_AOD_gap_mean,
    data_y=dataarray_CRF_sw_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[-100, -80, -60, -40],
    data_x_label="AOD",
    data_y_label="SWCRF",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="SWCRF",
    ymin=-160,
    ymax=-25,
    xmin=0.03,
    xmax=0.32,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-400,
    max=400,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)


####### IWP gap AOD-CRF_lw #############################

dataarray_sort_split = DataArrayPreprocessNoLog(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_CRF_lw,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean,
    dataarray_CRF_lw_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_CRF_lw_gap_mean
)
# intercept = np.exp(intercept)

# spatial split of linear regression
(
    slope_spatial_split,
    p_spatial_split,
) = dataarray_sort_split.dataarray_spatial_split()

# ---- plot --------------------
plot_both_nolog_swlw(
    data_x=dataarray_AOD_gap_mean,
    data_y=dataarray_CRF_lw_gap_mean,
    main_gap=IWP_gap,
    slope=slope,
    intercept=intercept,
    yticks=[0, 20, 40, 60,],
    data_x_label="AOD",
    data_y_label="LWCRF",
    main_gap_name="IWP",
    x_var_name="AOD",
    y_var_name="LWCRF",
    ymin=-50,
    ymax=65,
    xmin=0.03,
    xmax=0.32,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-400,
    max=400,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)


####### PC1 gap AOD-CRF_lw #############################

dataarray_sort_split = DataArrayPreprocessNoLog(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_CRF_lw,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean,
    dataarray_CRF_lw_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_CRF_lw_gap_mean,
)

# spatial split of linear regression
(
    slope_spatial_split,
    p_spatial_split,
) = dataarray_sort_split.dataarray_spatial_split()

# ---- plot --------------------
plot_both_nolog_swlw(
    data_x=dataarray_AOD_gap_mean,
    data_y=dataarray_CRF_lw_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[0, 20, 40,],
    data_x_label="AOD",
    data_y_label="LWCRF",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="LWCRF",
    ymin=-45,
    ymax=70,
    xmin=0.03,
    xmax=0.32,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-400,
    max=400,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)

# endregion

#

