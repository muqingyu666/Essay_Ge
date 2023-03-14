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
# Read in atmospheric data
data_atmos = read_in_data(
    "Data/EOF_T300_RH300_U300_w300_se300_lunkuonei.nc"
)
# Read in CLD data
data_CLD = read_in_data("Data/CERES_cloud.nc")
# Read in PC data
data_PC = read_in_data("Data/EOF_T300_RH300_U300_w300_se300.nc")

# ----- read in AOD\IWP\COD data -----
def extract_data(data_atmos, data_CLD, data_PC):
    """Extract relevant data from the atmospheric, cloud and PC data.

    Parameters
    ----------
    data_atmos : :obj:`dict`
        Dictionary containing the atmospheric data.
    data_CLD : :obj:`dict`
        Dictionary containing the cloud data.
    data_PC : :obj:`dict`
        Dictionary containing the principal component data.

    Returns
    -------
    data_temperture : :obj:`numpy.ndarray`
        Temperature data.
    data_humidity : :obj:`numpy.ndarray`
        Humidity data.
    data_wvelocity : :obj:`numpy.ndarray`
        Vertical velocity data.
    data_uwind : :obj:`numpy.ndarray`
        U wind data.
    data_unstability : :obj:`numpy.ndarray`
        Unstability data.
    data_PC : :obj:`numpy.ndarray`
        Principal component data.
    data_HCF : :obj:`numpy.ndarray`
        High cloud fraction data.

    """
    data_temperture = data_atmos["T_300_mask"]
    data_humidity = data_atmos["RH_300_mask"]
    data_wvelocity = data_atmos["w_300_mask"]
    data_uwind = data_atmos["U_300_mask"]
    data_unstability = data_atmos["dTdZ_300_mask"]
    data_PC = data_PC["EOF"][0, :, :, :]
    data_HCF = data_CLD["cldarea_high_daily"]

    return (
        data_temperture,
        data_humidity,
        data_wvelocity,
        data_uwind,
        data_unstability,
        data_PC,
        data_HCF,
    )


(
    data_temperture,
    data_humidity,
    data_wvelocity,
    data_uwind,
    data_unstability,
    data_PC,
    data_HCF,
) = extract_data(data_atmos, data_CLD, data_PC)

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


# Flatten the data
def flatten_data(data):
    # Function: flatten_data
    # Purpose: Flatten the data into an array of values
    # Input: data
    # Output: flattened data
    return data.values.flatten()


data_HCF = flatten_data(select_region(data_HCF))
data_PC = flatten_data(select_region(data_PC))
data_temperture = flatten_data(select_region(data_temperture))
data_humidity = flatten_data(select_region(data_humidity))
data_wvelocity = flatten_data(select_region(data_wvelocity))
data_uwind = flatten_data(select_region(data_uwind))
data_unstability = flatten_data(select_region(data_unstability))

# for the PC data woth nan, set the symtinous
# AOD, IWP, COD data to nan
# now the data are all 1d array
def give_nan_value(
    data_PC,
    data_HCF,
    data_temperture,
    data_humidity,
    data_wvelocity,
    data_uwind,
    data_unstability,
):
    data_HCF[np.isnan(data_PC)] = np.nan
    data_temperture[np.isnan(data_PC)] = np.nan
    data_humidity[np.isnan(data_PC)] = np.nan
    data_wvelocity[np.isnan(data_PC)] = np.nan
    data_uwind[np.isnan(data_PC)] = np.nan
    data_unstability[np.isnan(data_PC)] = np.nan


give_nan_value(
    data_PC,
    data_HCF,
    data_temperture,
    data_humidity,
    data_wvelocity,
    data_uwind,
    data_unstability,
)

data_PC = data_PC.reshape(4018, 17, 37)
data_HCF = data_HCF.reshape(4018, 17, 37)
data_temperture = data_temperture.reshape(4018, 17, 37)
data_humidity = data_humidity.reshape(4018, 17, 37)
data_wvelocity = data_wvelocity.reshape(4018, 17, 37)
data_uwind = data_uwind.reshape(4018, 17, 37)
data_unstability = data_unstability.reshape(4018, 17, 37)

####################################################
####################################################
##### DataArrayPreprocess class ####################
###################################################

# -----------------------------------------------
# --- calculate the correlation coefficient -----
# -----------------------------------------------
def calc_corr_coeff(dataarray_main, dataarray_aux):
    # calculate the correlation coefficient
    # between the main data and the auxiliary data
    # Input data may contain nan
    corr_coeff = np.empty((17, 37))
    p_value = np.empty((17, 37))

    for lat in range(17):
        for lon in range(37):
            # if the array value excpet nan is less than 3, set the corr_coeff and p_value to nan
            if (
                len(
                    dataarray_main[:, lat, lon][
                        ~np.isnan(dataarray_main[:, lat, lon])
                    ]
                )
                < 3
            ):
                corr_coeff[lat, lon] = np.nan
                p_value[lat, lon] = np.nan
            else:
                (
                    corr_coeff[lat, lon],
                    p_value[lat, lon],
                ) = stats.pearsonr(
                    dataarray_main[:, lat, lon][
                        ~np.isnan(dataarray_main[:, lat, lon])
                    ],
                    dataarray_aux[:, lat, lon][
                        ~np.isnan(dataarray_main[:, lat, lon])
                    ],
                )
    return corr_coeff, p_value


corr_coeff_temperture_HCF, p_value_temperture_HCF = calc_corr_coeff(
    data_temperture, data_HCF
)
corr_coeff_humidity_HCF, p_value_humidity_HCF = calc_corr_coeff(
    data_humidity, data_HCF
)
corr_coeff_wvelocity_HCF, p_value_wvelocity_HCF = calc_corr_coeff(
    data_wvelocity, data_HCF
)
corr_coeff_uwind_HCF, p_value_uwind_HCF = calc_corr_coeff(
    data_uwind, data_HCF
)
(
    corr_coeff_unstability_HCF,
    p_value_unstability_HCF,
) = calc_corr_coeff(data_unstability, data_HCF)

# combine the correlation coefficient and p_value
corr_all = np.stack(
    (
        corr_coeff_temperture_HCF,
        corr_coeff_humidity_HCF,
        corr_coeff_wvelocity_HCF,
        corr_coeff_uwind_HCF,
        corr_coeff_unstability_HCF,
    ),
    axis=0,
)
p_all = np.stack(
    (
        p_value_temperture_HCF,
        p_value_humidity_HCF,
        p_value_wvelocity_HCF,
        p_value_uwind_HCF,
        p_value_unstability_HCF,
    ),
    axis=0,
)

# -----------------------------------------------
# ---- divide the data into PC parts -------------
# -----------------------------------------------
class DataArrayCorrPreprocess:
    def __init__(self, dataarray_pc, dataarray_var, n):
        """
        initialize the class

        Args:
            dataarray_pc (np.array): the dataarray of PC
            dataarray_var (np.array): the dataarray of the variable
            n (int): the number of pieces to divide the dataarray
        """
        self.dataarray_pc = dataarray_pc.flatten()
        self.dataarray_var = dataarray_var.flatten()
        self.n = n

    def main_gap(self,):
        """
        Get the gap between each main piece

        Returns:
            main_gap_num: the gap between each main piece
        """
        PC_gap_num = np.zeros((self.n + 1))

        for i in range(self.n + 1):
            PC_gap_num[i] = np.nanpercentile(
                self.dataarray_pc, i * 100 / self.n
            )
        return PC_gap_num

    def dataarray_sort_split(self,):
        """
        Split dataarray into n pieces

        """
        PC_gap_num = self.main_gap()

        # get the gap between each main piece

        # filter the AOD, IWP, COD data by IWP gap
        # create empty array to store the data
        dataarray_var_gap_mean = np.empty((self.n))

        PC_gap_num_1 = np.roll(PC_gap_num, -1)
        PC_gap_mean = (
            PC_gap_num + PC_gap_num_1
        ) / 2.0  # これは長さ n の ndarray で最後の要素 (x[0]+x[n-1])/2. は要らない
        PC_gap_mean = PC_gap_mean[:-1]

        # main loop
        for i in range(self.n):
            # get the mean of the dataarray satisfying
            # the PC gap condition
            dataarray_var_gap_mean[i] = np.nanmean(
                self.dataarray_var[
                    np.where(
                        (self.dataarray_pc < PC_gap_num[i + 1])
                        & (self.dataarray_pc > PC_gap_num[i]),
                    )
                ]
            )

        return PC_gap_mean, dataarray_var_gap_mean


preprocess = DataArrayCorrPreprocess(data_PC, data_temperture, 32)
(
    PC_gap_mean,
    dataarray_temperture_gap_mean,
) = preprocess.dataarray_sort_split()

preprocess = DataArrayCorrPreprocess(data_PC, data_humidity, 32)
(
    PC_gap_mean,
    dataarray_humidity_gap_mean,
) = preprocess.dataarray_sort_split()

preprocess = DataArrayCorrPreprocess(data_PC, data_wvelocity, 32)
(
    PC_gap_mean,
    dataarray_wvelocity_gap_mean,
) = preprocess.dataarray_sort_split()

preprocess = DataArrayCorrPreprocess(data_PC, data_uwind, 32)
(
    PC_gap_mean,
    dataarray_uwind_gap_mean,
) = preprocess.dataarray_sort_split()

preprocess = DataArrayCorrPreprocess(data_PC, data_unstability, 32)
(
    PC_gap_mean,
    dataarray_unstability_gap_mean,
) = preprocess.dataarray_sort_split()

var_gap_mean = np.stack(
    (
        dataarray_temperture_gap_mean,
        dataarray_humidity_gap_mean,
        dataarray_wvelocity_gap_mean,
        dataarray_uwind_gap_mean,
        dataarray_unstability_gap_mean,
    ),
    axis=0,
)


def plot_5var_corr_p_line(
    corr, p, pc_gap_mean, var_gap_mean,
):
    import matplotlib.ticker as mticker

    # fig lst for subplots
    fig_lst = ["(a)", "(b)", "(c)", "(d)"]
    var_list = ["T", "RH", "W", "U", "Unstability"]

    # read in shapefile
    states_shp = Reader("DBATP/DBATP_Line.shp")

    # set up lat lon
    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    cmap = "RdBu_r"

    fig_lst = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
        "(i)",
        "(j)",
        "(k)",
    ]

    var_name = [
        "Temperature (K)",
        "Relative humidity (%)",
        "W velocity (m/s)",
        "U wind (m/s)",
        "Unstability (K/km)",
    ]

    fig = plt.figure(figsize=(8.8, 10), layout="constrained")

    spec = fig.add_gridspec(
        5,
        2,
        hspace=0.01,
        wspace=0.01,
        width_ratios=[1.2, 1],
        height_ratios=[1.1, 1.1, 1.1, 1.1, 1.1],
    )

    for i in range(5):
        ax = []
        ax.append(
            fig.add_subplot(
                spec[i, 0], projection=ccrs.PlateCarree()
            )
        )
        ax[0].set_extent([70, 105, 25, 40], crs=ccrs.PlateCarree())
        # axs[i].set_facecolor("silver")

        a = ax[0].pcolormesh(
            lon,
            lat,
            corr[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=-1,
            vmax=1,
        )

        gl = ax[0].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.6,
            alpha=0.8,
            draw_labels=True,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])

        ax[0].text(
            0.015,
            0.88,
            fig_lst[i * 2],
            transform=ax[0].transAxes,
            size=15,
            weight="bold",
        )

        ax[0].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p[i, :, :] < 0.05)

        dot = ax[0].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

        if i < 4:
            gl.bottom_labels = False
        else:
            cb = plt.colorbar(
                a,
                ax=ax,
                location="bottom",
                shrink=0.93,
                # pad=0.08,
                aspect=45,
            )

        ax = fig.add_subplot(spec[i, 1])
        ax.scatter(
            pc_gap_mean[1:-1],
            var_gap_mean[i, 1:-1],
            marker="o",
            color="k",
            alpha=0.5,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel(var_name[i])
        # disable the left label and show the right label
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xlim(-1.7, 1.2)
        # ax.grid(True, linestyle="-.", linewidth=0.6, alpha=0.8)
        if i == 2:
            ax.set_ylim(
                np.nanmin(var_gap_mean[i, :]) * 0.95,
                np.nanmax(var_gap_mean[i, :]) * 1.05,
            )
        if i < 4:
            ax.xaxis.set_ticks_position("none")
            ax.xaxis.set_ticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel("")

        ax.text(
            0.015,
            0.88,
            fig_lst[i * 2 + 1],
            size=15,
            transform=ax.transAxes,
            weight="bold",
        )

    plt.savefig(
        "corr_p_line.png", dpi=300, facecolor="w", edgecolor="w"
    )
    # plt.subplots_adjust(
    #     left=0.1,
    #     bottom=0.1,
    #     right=0.9,
    #     top=0.9,
    #     wspace=0.15,
    #     hspace=0.01,
    # )


plot_5var_corr_p_line(
    corr_all, p_all, PC_gap_mean, var_gap_mean,
)

