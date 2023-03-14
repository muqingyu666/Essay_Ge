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
    Mainly for AOD, IWP, COD data spatial distribution
    
    The util function is in muqy_20221224_util_Essay_plot_func.py
    
    Owner: Mu Qingyu
    version 1.0
          
    Created: 2023-02-10
    
    Including the following parts:

        1) Read in HCF、COD、PRE、IPR、IWP、CRF data
                
        2) Plot the spatial distribution of HCF、COD、PRE、IPR、IWP、CRF data
                
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


# # Flatten the data
def flatten_data(data):
    # Function: flatten_data
    # Purpose: Flatten the data into an array of values
    # Input: data
    # Output: flattened data
    return data.values.flatten()


# Flatten the data in order to filter nan values
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

# reshape the data into original shape
data_AOD = data_AOD.reshape(4018, 17, 37)
data_IWP = data_IWP.reshape(4018, 17, 37)
data_COD = data_COD.reshape(4018, 17, 37)
data_PC = data_PC.reshape(4018, 17, 37)
data_PRE = data_PRE.reshape(4018, 17, 37)
data_IPR = data_IPR.reshape(4018, 17, 37)
data_HCF = data_HCF.reshape(4018, 17, 37)
data_CRF = data_CRF.reshape(4018, 17, 37)
data_CRF_sw = data_CRF_sw.reshape(4018, 17, 37)
data_CRF_lw = data_CRF_lw.reshape(4018, 17, 37)

########################################################
#### Plot spatial distribution of cloud properties #####
########################################################


class YearlyMean(object):
    def __init__(self, data_array):
        self.data_array = data_array

    def each_year_mean(self):
        """divide the data into 11years from 2010-2020 and calculate the mean of each year"""
        data_2010 = self.data_array[0:365, :, :]
        data_2011 = self.data_array[365:730, :, :]
        data_2012 = self.data_array[730:1095, :, :]
        data_2013 = self.data_array[1095:1460, :, :]
        data_2014 = self.data_array[1460:1825, :, :]
        data_2015 = self.data_array[1825:2190, :, :]
        data_2016 = self.data_array[2190:2555, :, :]
        data_2017 = self.data_array[2555:2920, :, :]
        data_2018 = self.data_array[2920:3285, :, :]
        data_2019 = self.data_array[3285:3650, :, :]
        data_2020 = self.data_array[3650:, :, :]
        data_2010_mean = np.nanmean(data_2010, axis=0)
        data_2011_mean = np.nanmean(data_2011, axis=0)
        data_2012_mean = np.nanmean(data_2012, axis=0)
        data_2013_mean = np.nanmean(data_2013, axis=0)
        data_2014_mean = np.nanmean(data_2014, axis=0)
        data_2015_mean = np.nanmean(data_2015, axis=0)
        data_2016_mean = np.nanmean(data_2016, axis=0)
        data_2017_mean = np.nanmean(data_2017, axis=0)
        data_2018_mean = np.nanmean(data_2018, axis=0)
        data_2019_mean = np.nanmean(data_2019, axis=0)
        data_2020_mean = np.nanmean(data_2020, axis=0)
        return np.concatenate(
            (
                data_2010_mean.reshape(1, 17, 37),
                data_2011_mean.reshape(1, 17, 37),
                data_2012_mean.reshape(1, 17, 37),
                data_2013_mean.reshape(1, 17, 37),
                data_2014_mean.reshape(1, 17, 37),
                data_2015_mean.reshape(1, 17, 37),
                data_2016_mean.reshape(1, 17, 37),
                data_2017_mean.reshape(1, 17, 37),
                data_2018_mean.reshape(1, 17, 37),
                data_2019_mean.reshape(1, 17, 37),
                data_2020_mean.reshape(1, 17, 37),
            ),
            axis=0,
        )

    def all_yearly_mean(self):
        """calculate the mean of all years"""
        all_mean_data = np.nanmean(self.each_year_mean(), axis=0)
        return all_mean_data


ym_AOD = YearlyMean(data_AOD)
all_mean_AOD = ym_AOD.all_yearly_mean()

ym_IWP = YearlyMean(data_IWP)
all_mean_IWP = ym_IWP.all_yearly_mean()

ym_COD = YearlyMean(data_COD)
all_mean_COD = ym_COD.all_yearly_mean()

ym_PC = YearlyMean(data_PC)
all_mean_PC = ym_PC.all_yearly_mean()

ym_PRE = YearlyMean(data_PRE)
all_mean_PRE = ym_PRE.all_yearly_mean()

ym_IPR = YearlyMean(data_IPR)
all_mean_IPR = ym_IPR.all_yearly_mean()

ym_HCF = YearlyMean(data_HCF)
all_mean_HCF = ym_HCF.all_yearly_mean()

ym_CRF = YearlyMean(data_CRF)
all_mean_CRF = ym_CRF.all_yearly_mean()

ym_CRF_sw = YearlyMean(data_CRF_sw)
all_mean_CRF_sw = ym_CRF_sw.all_yearly_mean()

ym_CRF_lw = YearlyMean(data_CRF_lw)
all_mean_CRF_lw = ym_CRF_lw.all_yearly_mean()

# plot each variales in each figure
plot_spatial_distribution_of_var(
    dataarray=all_mean_AOD,
    min=0,
    max=0.4,
    var_name="AOD",
    cmap_file="Color/PC1_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_IWP,
    min=0,
    max=320,
    var_name="IWP",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_COD,
    min=0,
    max=7,
    var_name="COD",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_PC,
    min=0,
    max=0.2,
    var_name="PC",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_PRE,
    min=0,
    max=11,
    var_name="PRE",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_IPR,
    min=25,
    max=34,
    var_name="IPR",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_HCF,
    min=0,
    max=40,
    var_name="HCF",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_CRF,
    min=0,
    max=100,
    var_name="CRF",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_CRF_sw,
    min=-100,
    max=0,
    var_name="CRF_sw",
    cmap_file="Color/Var_color.txt",
)
plot_spatial_distribution_of_var(
    dataarray=all_mean_CRF_lw,
    min=0,
    max=100,
    var_name="CRF_lw",
    cmap_file="Color/Var_color.txt",
)

# concatenate 6 variables
var_all = np.stack(
    (
        all_mean_HCF,
        all_mean_PRE,
        all_mean_COD,
        all_mean_IWP,
        all_mean_IPR,
        all_mean_CRF,
    ),
    axis=0,
)
var_name_all = [
    "HCF (%)",
    "PRE (mm/day)",
    "COD",
    "IWP (g/m" + r"$^2$" + ")",
    "IPR (" + r"$\mu$" + r"m)",
    "CRF (W/m" + r"$^2$" + ")",
]
var_cbar_ticks_all = [
    np.round(np.linspace(11, 32, num=4)).tolist(),
    np.round(np.linspace(0, 9, num=4)).tolist(),
    np.round(np.linspace(1, 4, num=4)).tolist(),
    np.round(np.linspace(80, 195, num=4)).tolist(),
    np.round(np.linspace(26, 31, num=4)).tolist(),
    np.round(np.linspace(0, 66, num=4)).tolist(),
]
var_min_all = [9, -1.5, 0.4, 65, 25.5, -8]
var_max_all = [34, 10, 4.7, 210, 32, 74]

# plot 6 variables in one figure
# HCF、COD、PRE、IPR、IWP、CRF
plot_spatial_distribution_of_6var_fig(
    var_dataarray=var_all,
    min=var_min_all,
    max=var_max_all,
    var_name=var_name_all,
    var_cbar_ticks=var_cbar_ticks_all,
    cmap_file="Color/Var_color.txt",
)
