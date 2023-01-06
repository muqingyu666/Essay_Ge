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
data_AER = read_in_data("Data/AOD_bilinear_2010-2020.nc")
data_CLD = read_in_data("Data/CERES_cloud.nc")
data_PC = read_in_data("Data/EOF_T300_RH300_U300_w300_se300.nc")
data_PRE = read_in_data("Data/GPM_IMGER_2010-2020.nc")

# ----- read in AOD\IWP\COD data -----
data_AOD = data_AER["AOD_bilinear"][:, ::-1, :]
data_IWP = data_CLD["iwp_high_daily"]
data_IPR = data_CLD["cldicerad_high_daily"]
data_HCF = data_CLD["cldarea_high_daily"]
data_COD = data_CLD["cldtau_high_daily"]
data_PC = data_PC["EOF"][0, :, :, :]
data_PRE = data_PRE["preicp"]


# specify the region
latS = 24.5
latN = 40.5
lonL = 69.5
lonR = 105.5

data_AOD = data_AOD.sel(
    lat=slice(latS, latN), lon=slice(lonL, lonR)
)
data_IWP = data_IWP.sel(
    lat=slice(latS, latN), lon=slice(lonL, lonR)
)
data_COD = data_COD.sel(
    lat=slice(latS, latN), lon=slice(lonL, lonR)
)
data_PRE = data_PRE.sel(
    lat=slice(latS, latN), lon=slice(lonL, lonR)
)
data_IPR = data_IPR.sel(
    lat=slice(latS, latN), lon=slice(lonL, lonR)
)
data_HCF = data_HCF.sel(
    lat=slice(latS, latN), lon=slice(lonL, lonR)
)

# flatten the data
data_AOD = data_AOD.values.flatten()
data_IWP = data_IWP.values.flatten()
data_COD = data_COD.values.flatten()
data_PC = data_PC.values.flatten()
data_PRE = data_PRE.values.flatten()
data_IPR = data_IPR.values.flatten()
data_HCF = data_HCF.values.flatten()

# this is critical, only filter the data with AOD < 0.06
# can reproduce the results in the paper
data_AOD[data_AOD < 0.06] = np.nan
data_PRE[data_PRE == 0] = np.nan

# for the PC data woth nan, set the symtinous
# AOD, IWP, COD data to nan
# now the data are all 1d array
data_AOD[np.isnan(data_PC)] = np.nan
data_IWP[np.isnan(data_PC)] = np.nan
data_COD[np.isnan(data_PC)] = np.nan
data_PRE[np.isnan(data_PC)] = np.nan
data_IPR[np.isnan(data_PC)] = np.nan
data_HCF[np.isnan(data_PC)] = np.nan

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
####################################################
####################################################
##### DataArrayPreprocess class ####################
###3###############################################

######## IWP gap AOD-COD ############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_COD,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean,
    dataarray_COD_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_COD_gap_mean
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
    data_y=dataarray_COD_gap_mean,
    main_gap=IWP_gap,
    slope=slope,
    intercept=intercept,
    yticks=[-2, -1, 0, 1, 2],
    data_x_label="ln(AOD)",
    data_y_label="ln(COD)",
    main_gap_name="IWP",
    x_var_name="AOD",
    y_var_name="COD",
    ymin=-5.8,
    ymax=2.4,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-0.94,
    max=0.94,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)


####### PC gap AOD-COD #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_COD,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean,
    dataarray_COD_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_COD_gap_mean
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
    data_y=dataarray_COD_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[-1, 0, 1, 2],
    data_x_label="ln(AOD)",
    data_y_label="ln(COD)",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="COD",
    ymin=-4.1,
    ymax=2.2,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-1.7,
    max=1.7,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)

####### IWP gap AOD-IPR #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_IWP,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_IPR,
    n=6,
)

# get data of line plot
(
    IWP_gap,
    dataarray_AOD_gap_mean,
    dataarray_IPR_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_IPR_gap_mean
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
    data_y=dataarray_IPR_gap_mean,
    main_gap=IWP_gap,
    slope=slope,
    intercept=intercept,
    yticks=[3.25, 3.3, 3.35, 3.4, 3.45],
    data_x_label="ln(AOD)",
    data_y_label="ln(IPR)",
    main_gap_name="IWP",
    x_var_name="AOD",
    y_var_name="IPR",
    ymin=3.0,
    ymax=3.47,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-0.2,
    max=0.2,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)

####### PC1 gap AOD-IPR #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_IPR,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean,
    dataarray_IPR_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_IPR_gap_mean
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
    data_y=dataarray_IPR_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[3.34, 3.36, 3.38],
    data_x_label="ln(AOD)",
    data_y_label="ln(IPR)",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="IPR",
    ymin=3.28,
    ymax=3.395,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-0.2,
    max=0.2,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
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
    dataarray_AOD_gap_mean,
    dataarray_HCF_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_HCF_gap_mean
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
    data_y=dataarray_HCF_gap_mean,
    main_gap=IWP_gap,
    slope=slope,
    intercept=intercept,
    yticks=[0, 1, 2, 3],
    data_x_label="ln(AOD)",
    data_y_label="ln(HCF)",
    main_gap_name="IWP",
    x_var_name="AOD",
    y_var_name="HCF",
    ymin=-2.6,
    ymax=3.5,
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

####### PC1 gap AOD-HCF #############################

dataarray_sort_split = DataArrayPreprocess(
    dataarray_main=data_PC,
    dataarray_aux0=data_AOD,
    dataarray_aux1=data_HCF,
    n=6,
)

# get data of line plot
(
    PC_gap,
    dataarray_AOD_gap_mean,
    dataarray_HCF_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_HCF_gap_mean
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
    data_y=dataarray_HCF_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[0, 1, 2, 3],
    data_x_label="ln(AOD)",
    data_y_label="ln(HCF)",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="HCF",
    ymin=-2.6,
    ymax=3.9,
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
    dataarray_AOD_gap_mean,
    dataarray_PRE_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_PRE_gap_mean
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
    data_y=dataarray_PRE_gap_mean,
    main_gap=IWP_gap,
    slope=slope,
    intercept=intercept,
    yticks=[-3, -2, -1, 0, 1],
    data_x_label="ln(AOD)",
    data_y_label="ln(PRE)",
    main_gap_name="IWP",
    x_var_name="AOD",
    y_var_name="PRE",
    ymin=-7,
    ymax=1.1,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-3,
    max=3,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)


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
    dataarray_AOD_gap_mean,
    dataarray_PRE_gap_mean,
) = dataarray_sort_split.dataarray_sort_split()


# linear regression of line plot
(
    slope,
    intercept,
    p_value,
) = dataarray_sort_split.linear_regression_lst(
    dataarray_AOD_gap_mean, dataarray_PRE_gap_mean
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
    data_y=dataarray_PRE_gap_mean,
    main_gap=PC_gap,
    slope=slope,
    intercept=intercept,
    yticks=[-3, -2, -1, 0, 1],
    data_x_label="ln(AOD)",
    data_y_label="ln(PRE)",
    main_gap_name="PC1",
    x_var_name="AOD",
    y_var_name="PRE",
    ymin=-7,
    ymax=1.1,
    xmin=-2.9,
    xmax=-1.1,
    linewidth=3,
    dataarray=slope_spatial_split,
    p_spatial_split=p_spatial_split,
    min=-3.2,
    max=3.2,
    var_name="slope",
    cmap_file="Color/HCF_color.txt",
)

#########################################
# expired code ##########################
# plot_line(
#     dataarray_AOD_gap_mean,
#     dataarray_COD_gap_mean,
#     "ln(AOD)",
#     "ln(COD)",
#     ymin=-4.8,
#     ymax=2.2,
#     xmin=-2.9,
#     xmax=-1.1,
#     linewidth=3,
# )


# plot_spatial_split(
#     slope_spatial_split,
#     -1.5,
#     1.5,
#     "slope",
#     cmap_file="Color/HCF_color.txt",
# )
