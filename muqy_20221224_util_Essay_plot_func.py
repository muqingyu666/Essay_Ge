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

    Code for gejinming's essay plot
    Util functions
        
    Owner: Mu Qingyu
    version 1.0
          
    Created: 2022-12-24
    
    Including the following parts:

        1) Data preprocess module
                
        2) Plot module
        
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

# import done

# set the font and style
mpl.rcParams["font.family"] = "sans-serif"
mpl.style.use("seaborn-ticks")


###############################################
# ----- read in data module ----- #############
###############################################


def read_in_data(filename):
    """
    Read in data from netcdf file

    Return:
        data: xarray dataset
    """
    data = xr.open_dataset(filename)
    return data


###############################################
## ----- data preprocess module ----- #########
###############################################

class DataArrayPreprocess:
    def __init__(
        self, dataarray_main, dataarray_aux0, dataarray_aux1, n
    ):
        """
        Initializes the class.
        """
        self.dataarray_main = dataarray_main
        self.dataarray_aux0 = dataarray_aux0
        self.dataarray_aux1 = dataarray_aux1
        self.n = n

    def main_gap(self,):
        """
        Get the gap between each main piece

        Returns:
            main_gap_num: the gap between each main piece
        """
        main_gap_num = np.empty((self.n + 1))

        for i in range(self.n + 1):
            main_gap_num[i] = np.nanpercentile(
                self.dataarray_main, i * 100 / self.n
            )
        return np.round(main_gap_num, 2)

    def dataarray_sort_split(self,):
        """
        Split dataarray into n pieces

        """

        main_gap_num = self.main_gap()

        # get the gap between each main piece

        # filter the AOD, IWP, COD data by IWP gap
        # create empty array to store the data
        dataarray_aux0_gap_mean = np.empty((self.n, self.n))
        dataarray_aux1_gap_mean = np.empty((self.n, self.n))

        # use log to make the data more linear
        dataarray_aux0 = np.log(self.dataarray_aux0)
        dataarray_aux1 = np.log(self.dataarray_aux1)

        # get the gap between each aux0 piece
        aux0_gap_num = np.empty((self.n + 1))

        for i in range(self.n + 1):
            aux0_gap_num[i] = np.nanpercentile(
                dataarray_aux0, i * 100 / self.n
            )

        # main loop
        for i in range(self.n):
            for j in range(self.n):

                dataarray_aux0_gap_mean[i, j] = np.nanmean(
                    dataarray_aux0[
                        np.where(
                            (dataarray_aux0 < aux0_gap_num[i + 1])
                            & (dataarray_aux0 > aux0_gap_num[i])
                            & (
                                self.dataarray_main
                                < main_gap_num[j + 1]
                            )
                            & (
                                self.dataarray_main
                                > main_gap_num[j]
                            ),
                        )
                    ]
                )

                dataarray_aux1_gap_mean[i, j] = np.nanmean(
                    dataarray_aux1[
                        np.where(
                            (dataarray_aux0 < aux0_gap_num[i + 1])
                            & (dataarray_aux0 > aux0_gap_num[i])
                            & (
                                self.dataarray_main
                                < main_gap_num[j + 1]
                            )
                            & (
                                self.dataarray_main
                                > main_gap_num[j]
                            ),
                        )
                    ]
                )

        return (
            np.round(main_gap_num, 1),
            dataarray_aux0_gap_mean.transpose(),
            dataarray_aux1_gap_mean.transpose(),
        )

    def dataarray_spatial_split(self,):
        """
        Split dataarray into spatial pieces
        """
        main_gap_num = self.main_gap()

        # get the gap between each main piece
        dataarray_aux0_spatial_split = np.empty((4, 4018, 17, 37))
        dataarray_aux1_spatial_split = np.empty((4, 4018, 17, 37))

        dataarray_aux0_spatial_split[0] = np.where(
            (self.dataarray_main < main_gap_num[1])
            & (self.dataarray_main > main_gap_num[0]),
            np.log(self.dataarray_aux0),
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[0] = np.where(
            (self.dataarray_main < main_gap_num[1])
            & (self.dataarray_main > main_gap_num[0]),
            np.log(self.dataarray_aux1),
            np.nan,
        ).reshape(4018, 17, 37)

        dataarray_aux0_spatial_split[1] = np.where(
            (self.dataarray_main < main_gap_num[3])
            & (self.dataarray_main > main_gap_num[1]),
            np.log(self.dataarray_aux0),
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[1] = np.where(
            (self.dataarray_main < main_gap_num[3])
            & (self.dataarray_main > main_gap_num[1]),
            np.log(self.dataarray_aux1),
            np.nan,
        ).reshape(4018, 17, 37)

        dataarray_aux0_spatial_split[2] = np.where(
            (self.dataarray_main < main_gap_num[5])
            & (self.dataarray_main > main_gap_num[3]),
            np.log(self.dataarray_aux0),
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[2] = np.where(
            (self.dataarray_main < main_gap_num[5])
            & (self.dataarray_main > main_gap_num[3]),
            np.log(self.dataarray_aux1),
            np.nan,
        ).reshape(4018, 17, 37)

        dataarray_aux0_spatial_split[3] = np.where(
            (self.dataarray_main < main_gap_num[6])
            & (self.dataarray_main > main_gap_num[5]),
            np.log(self.dataarray_aux0),
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[3] = np.where(
            (self.dataarray_main < main_gap_num[6])
            & (self.dataarray_main > main_gap_num[5]),
            np.log(self.dataarray_aux1),
            np.nan,
        ).reshape(4018, 17, 37)

        # main loop
        slope_spatial_split = np.empty((4, 17, 37))
        p_spatial_split = np.empty((4, 17, 37))

        for gap in range(4):
            for lat in range(17):
                for lon in range(37):

                    if (
                        dataarray_aux1_spatial_split[
                            gap, :, lat, lon
                        ][
                            ~np.isnan(
                                dataarray_aux1_spatial_split[
                                    gap, :, lat, lon
                                ]
                            )
                        ].shape[
                            0
                        ]
                        >= 3
                    ):
                        mask = ~np.isnan(
                            dataarray_aux0_spatial_split[
                                gap, :, lat, lon
                            ]
                        ) & ~np.isnan(
                            dataarray_aux1_spatial_split[
                                gap, :, lat, lon
                            ]
                        )

                        (
                            slope_spatial_split[gap, lat, lon],
                            _,
                            p_spatial_split[gap, lat, lon],
                        ) = self.linear_regression(
                            dataarray_aux0_spatial_split[
                                gap, :, lat, lon
                            ][mask],
                            dataarray_aux1_spatial_split[
                                gap, :, lat, lon
                            ][mask],
                        )

                    else:
                        slope_spatial_split[gap, lat, lon] = np.nan
                        p_spatial_split[gap, lat, lon] = np.nan

        # set grid point which do not pass the significance test to nan
        p_spatial_split[p_spatial_split > 0.05] = np.nan

        return slope_spatial_split, p_spatial_split

    def linear_regression(self, data_x, data_y):
        """
        linear regression
        """

        (
            slope,
            intercept,
            r_value,
            p_value,
            std_err,
        ) = stats.linregress(data_x, data_y)

        return slope, intercept, p_value

    def linear_regression_lst(self, data_x_lst, data_y_lst):
        """
        linear regression
        """

        slope_lst = []
        intercept_lst = []
        p_value_lst = []

        for i in range(self.n):
            (slope, intercept, p_value,) = self.linear_regression(
                data_x_lst[i], data_y_lst[i]
            )
            slope_lst.append(slope)
            intercept_lst.append(intercept)
            p_value_lst.append(p_value)

        return slope_lst, intercept_lst, p_value_lst


class DataArrayPreprocessNoLog:
    def __init__(
        self, dataarray_main, dataarray_aux0, dataarray_aux1, n
    ):
        self.dataarray_main = dataarray_main
        self.dataarray_aux0 = dataarray_aux0
        self.dataarray_aux1 = dataarray_aux1
        self.n = n

    def main_gap(self,):
        """
        Get the gap between each main piece

        Returns:
            main_gap_num: the gap between each main piece
        """
        main_gap_num = np.empty((self.n + 1))

        for i in range(self.n + 1):
            main_gap_num[i] = np.nanpercentile(
                self.dataarray_main, i * 100 / self.n
            )
        return main_gap_num

    def dataarray_sort_split(self,):
        """
        Split dataarray into n pieces

        """

        main_gap_num = self.main_gap()

        # get the gap between each main piece

        # filter the AOD, IWP, COD data by IWP gap
        # create empty array to store the data
        dataarray_aux0_gap_mean = np.empty((self.n, self.n))
        dataarray_aux1_gap_mean = np.empty((self.n, self.n))

        # not use log to make the data more linear
        dataarray_aux0 = self.dataarray_aux0
        dataarray_aux1 = self.dataarray_aux1

        # get the gap between each aux0 piece
        aux0_gap_num = np.empty((self.n + 1))

        for i in range(self.n + 1):
            aux0_gap_num[i] = np.nanpercentile(
                dataarray_aux0, i * 100 / self.n
            )

        # main loop
        for i in range(self.n):
            for j in range(self.n):

                dataarray_aux0_gap_mean[i, j] = np.nanmean(
                    dataarray_aux0[
                        np.where(
                            (dataarray_aux0 < aux0_gap_num[i + 1])
                            & (dataarray_aux0 > aux0_gap_num[i])
                            & (
                                self.dataarray_main
                                < main_gap_num[j + 1]
                            )
                            & (
                                self.dataarray_main
                                > main_gap_num[j]
                            ),
                        )
                    ]
                )

                dataarray_aux1_gap_mean[i, j] = np.nanmean(
                    dataarray_aux1[
                        np.where(
                            (dataarray_aux0 < aux0_gap_num[i + 1])
                            & (dataarray_aux0 > aux0_gap_num[i])
                            & (
                                self.dataarray_main
                                < main_gap_num[j + 1]
                            )
                            & (
                                self.dataarray_main
                                > main_gap_num[j]
                            ),
                        )
                    ]
                )

        return (
            np.round(main_gap_num, 1),
            dataarray_aux0_gap_mean.transpose(),
            dataarray_aux1_gap_mean.transpose(),
        )

    def dataarray_spatial_split(self,):
        """
        Split dataarray into spatial pieces
        """
        main_gap_num = self.main_gap()

        # get the gap between each main piece
        dataarray_aux0_spatial_split = np.empty((4, 4018, 17, 37))
        dataarray_aux1_spatial_split = np.empty((4, 4018, 17, 37))

        dataarray_aux0_spatial_split[0] = np.where(
            (self.dataarray_main < main_gap_num[1])
            & (self.dataarray_main > main_gap_num[0]),
            self.dataarray_aux0,
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[0] = np.where(
            (self.dataarray_main < main_gap_num[1])
            & (self.dataarray_main > main_gap_num[0]),
            self.dataarray_aux1,
            np.nan,
        ).reshape(4018, 17, 37)

        dataarray_aux0_spatial_split[1] = np.where(
            (self.dataarray_main < main_gap_num[3])
            & (self.dataarray_main > main_gap_num[1]),
            self.dataarray_aux0,
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[1] = np.where(
            (self.dataarray_main < main_gap_num[3])
            & (self.dataarray_main > main_gap_num[1]),
            self.dataarray_aux1,
            np.nan,
        ).reshape(4018, 17, 37)

        dataarray_aux0_spatial_split[2] = np.where(
            (self.dataarray_main < main_gap_num[5])
            & (self.dataarray_main > main_gap_num[3]),
            self.dataarray_aux0,
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[2] = np.where(
            (self.dataarray_main < main_gap_num[5])
            & (self.dataarray_main > main_gap_num[3]),
            self.dataarray_aux1,
            np.nan,
        ).reshape(4018, 17, 37)

        dataarray_aux0_spatial_split[3] = np.where(
            (self.dataarray_main < main_gap_num[6])
            & (self.dataarray_main > main_gap_num[5]),
            self.dataarray_aux0,
            np.nan,
        ).reshape(4018, 17, 37)
        dataarray_aux1_spatial_split[3] = np.where(
            (self.dataarray_main < main_gap_num[6])
            & (self.dataarray_main > main_gap_num[5]),
            self.dataarray_aux1,
            np.nan,
        ).reshape(4018, 17, 37)

        # main loop
        slope_spatial_split = np.empty((4, 17, 37))
        p_spatial_split = np.empty((4, 17, 37))

        for gap in range(4):
            for lat in range(17):
                for lon in range(37):

                    if (
                        dataarray_aux1_spatial_split[
                            gap, :, lat, lon
                        ][
                            ~np.isnan(
                                dataarray_aux1_spatial_split[
                                    gap, :, lat, lon
                                ]
                            )
                        ].shape[
                            0
                        ]
                        >= 3
                    ):
                        mask = ~np.isnan(
                            dataarray_aux0_spatial_split[
                                gap, :, lat, lon
                            ]
                        ) & ~np.isnan(
                            dataarray_aux1_spatial_split[
                                gap, :, lat, lon
                            ]
                        )

                        (
                            slope_spatial_split[gap, lat, lon],
                            _,
                            p_spatial_split[gap, lat, lon],
                        ) = self.linear_regression(
                            dataarray_aux0_spatial_split[
                                gap, :, lat, lon
                            ][mask],
                            dataarray_aux1_spatial_split[
                                gap, :, lat, lon
                            ][mask],
                        )

                    else:
                        slope_spatial_split[gap, lat, lon] = np.nan
                        p_spatial_split[gap, lat, lon] = np.nan

        # set grid point which do not pass the significance test to nan
        p_spatial_split[p_spatial_split > 0.05] = np.nan

        return slope_spatial_split, p_spatial_split

    def linear_regression(self, data_x, data_y):
        """
        linear regression
        """

        (
            slope,
            intercept,
            r_value,
            p_value,
            std_err,
        ) = stats.linregress(data_x, data_y)

        return slope, intercept, p_value

    def linear_regression_lst(self, data_x_lst, data_y_lst):
        """
        linear regression
        """

        slope_lst = []
        intercept_lst = []
        p_value_lst = []

        for i in range(self.n):
            (slope, intercept, p_value,) = self.linear_regression(
                data_x_lst[i], data_y_lst[i]
            )
            slope_lst.append(slope)
            intercept_lst.append(intercept)
            p_value_lst.append(p_value)

        return slope_lst, intercept_lst, p_value_lst


############################################################################################################
#### ---- Plotting ---- ##############################################################################
############################################################################################################

# --- color map modoule--- #


def dcmap(file_path):
    fid = open(file_path)
    data = fid.readlines()
    n = len(data)
    rgb = np.empty((n, 3))
    for i in np.arange(n):
        rgb[i][0] = data[i].split(",")[0]
        rgb[i][1] = data[i].split(",")[1]
        rgb[i][2] = data[i].split(",")[2]
        rgb[i] = rgb[i] / 255.0
        icmap = mpl.colors.ListedColormap(rgb, name="my_color")
    return icmap


# --- spatial plot --- #


def plot_spatial_split(
    dataarray,
    p_spatial_split,
    min,
    max,
    var_name,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot spatial distribution of dataarray with spatial split.

    Args:
        dataarray (np.array): main dataarray of slope
        p_spatial_split (np.array): main dataarray of p value
        min (float): min value of colorbar
        max (float): max value of colorbar
        var_name (str): name of variable
        cmap_file (str, optional): cmap file directory. Defaults to "Color/PC1_color.txt".
    """
    # fig lst for subplots
    fig_lst = ["(a)", "(b)", "(c)", "(d)"]

    # read in shapefile
    states_shp = Reader("DBATP/DBATP_Line.shp")

    # set up lat lon
    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 7.5),
        # tight_layout=True,
        # constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    # plt.rcParams.update({"font.family": "Times New Roman"})

    axs = axs.flatten()

    for i in range(4):
        axs[i].set_extent([70, 105, 25, 40], crs=ccrs.PlateCarree())
        # axs[i].set_facecolor("silver")

        a = axs[i].pcolormesh(
            lon,
            lat,
            dataarray[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min,
            vmax=max,
        )

        gl = axs[i].gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        gl.left_labels = False

        axs[i].text(
            0.015,
            0.88,
            fig_lst[i],
            transform=axs[i].transAxes,
            size=15,
            weight="bold",
        )

        axs[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split[i] < 0.05)
        dot = axs[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.15,
        hspace=0.01,
    )

    cb = plt.colorbar(
        a,
        ax=axs,
        location="bottom",
        shrink=0.93,
        extend="both",
        pad=0.08,
        aspect=45,
    )

    # cb.set_label(label=var_name, size=24)

    # cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    # gl.xlabel_style = {"size": 18}
    # gl.ylabel_style = {"size": 18}


# ---- plot both ---- #


def plot_both(
    # line plot part
    data_x,
    data_y,
    main_gap,
    slope,
    intercept,
    yticks,
    data_x_label,
    data_y_label,
    main_gap_name,
    x_var_name,
    y_var_name,
    ymin,
    ymax,
    xmin,
    xmax,
    linewidth,
    # spatial plot part
    dataarray,
    p_spatial_split,
    min,
    max,
    var_name,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot both line plot and spatial plot.

    Args:
        data_x (np.array): x data for line plot
        data_y (np.array): y data for line plot
        main_gap (np.array): main gap for line plot
        slope (np.array): slope for linear regression
        intercept (np.array): intercept for linear regression
        data_x_label (str): x label for line plot
        data_y_label (str): y label for line plot
        ymin (float): min value of y axis for line plot
        ymax (float): max value of y axis for line plot
        xmin (float): min value of x axis for line plot
        xmax (float): max value of x axis for line plot
        linewidth (float): linewidth for line plot
        dataarray (np.array): data for spatial plot slope
        p_spatial_split (np.array): p value for spatial plot
        min (float): min value for colorbar
        max (float): max value for colorbar
        var_name (str): variable name for colorbar
        cmap_file (str, optional): file dir for cmap. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.gridspec as gridspec

    fig_lst = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
    ]

    fig = plt.figure(figsize=(17, 4.5), dpi=300)

    gs0 = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.1, 1.75]
    )

    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs0[0]
    )

    axbig = fig.add_subplot(gs00[0, 0])

    axbig.plot(
        data_x[0],
        data_y[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap[0])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[1]),
    )
    axbig.plot(
        data_x[1],
        data_y[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap[1])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[2]),
    )
    axbig.plot(
        data_x[2],
        data_y[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap[2])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[3]),
    )
    axbig.plot(
        data_x[3],
        data_y[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap[3])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[4]),
    )
    axbig.plot(
        data_x[4],
        data_y[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap[4])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[5]),
    )
    axbig.plot(
        data_x[5],
        data_y[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap[5])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin, ymax)
    axbig.set_xlim(xmin, xmax)
    axbig.set_yticks(yticks)

    axbig.legend(
        loc="lower left",
        handlelength=4.5,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.53,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept[0], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope[0], 3))
        + "}}$ "
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept[1], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope[1], 3))
        + "}}$ "
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept[2], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope[2], 3))
        + "}}$ "
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept[3], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope[3], 3))
        + "}}$ "
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept[4], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope[4], 3))
        + "}}$ "
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept[5], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope[5], 3))
        + "}}$ "
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.55, 0),
        fontsize=12,
        labelspacing=0.1,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs01[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min,
            vmax=max,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.88,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    plt.subplots_adjust(
        wspace=0.13, hspace=0.15,
    )

    cax = fig.add_axes([0.45, -0.02, 0.43, 0.03])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="horizontal",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )
    plt.savefig(
        "figure/"
        + data_x_label
        + "_"
        + data_y_label
        + "_"
        + main_gap_name
        + ".png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )


def plot_both_pc1_iwp(
    # figure part1
    # line plot part
    data_x_0,
    data_y_0,
    main_gap_0,
    slope_0,
    intercept_0,
    yticks_0,
    main_gap_name_0,
    ymin_0,
    ymax_0,
    xmin_0,
    xmax_0,
    # spatial plot part
    dataarray_0,
    p_spatial_split_0,
    vmin_0,
    vmax_0,
    # figure part2
    # line plot part
    data_x_1,
    data_y_1,
    main_gap_1,
    slope_1,
    intercept_1,
    yticks_1,
    main_gap_name_1,
    ymin_1,
    ymax_1,
    xmin_1,
    xmax_1,
    # spatial plot part
    dataarray_1,
    p_spatial_split_1,
    vmin_1,
    vmax_1,
    # Universal settings
    data_x_label,
    data_y_label,
    x_var_name,
    y_var_name,
    linewidth,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot both line plot and spatial plot.

    Args:
        data_x (np.array): x data for line plot
        data_y (np.array): y data for line plot
        main_gap (np.array): main gap for line plot
        slope (np.array): slope for linear regression
        intercept (np.array): intercept for linear regression
        data_x_label (str): x label for line plot
        data_y_label (str): y label for line plot
        ymin (float): min value of y axis for line plot
        ymax (float): max value of y axis for line plot
        xmin (float): min value of x axis for line plot
        xmax (float): max value of x axis for line plot
        linewidth (float): linewidth for line plot
        dataarray (np.array): data for spatial plot slope
        p_spatial_split (np.array): p value for spatial plot
        min (float): min value for colorbar
        max (float): max value for colorbar
        var_name (str): variable name for colorbar
        cmap_file (str, optional): file dir for cmap. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.gridspec as gridspec

    fig_lst = [
        "(a)",
        "(a1)",
        "(a2)",
        "(a3)",
        "(a4)",
        "(a5)",
        "(a6)",
        "(a7)",
    ]

    fig = plt.figure(figsize=(17, 9.6), dpi=300)

    gs_all = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, 1], hspace=0.2
    )

    # figure part1
    gs0 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_all[0], width_ratios=[1.1, 1.75]
    )

    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs0[0]
    )

    axbig = fig.add_subplot(gs00[0, 0])

    axbig.plot(
        data_x_0[0],
        data_y_0[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap_0[0])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[1]),
    )
    axbig.plot(
        data_x_0[1],
        data_y_0[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap_0[1])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[2]),
    )
    axbig.plot(
        data_x_0[2],
        data_y_0[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap_0[2])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[3]),
    )
    axbig.plot(
        data_x_0[3],
        data_y_0[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap_0[3])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[4]),
    )
    axbig.plot(
        data_x_0[4],
        data_y_0[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap_0[4])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[5]),
    )
    axbig.plot(
        data_x_0[5],
        data_y_0[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap_0[5])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin_0, ymax_0)
    axbig.set_xlim(xmin_0, xmax_0)
    axbig.set_yticks(yticks_0)

    axbig.legend(
        loc="lower left",
        handlelength=4.5,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.53,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_0[0], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_0[0], 3))
        + "}}$ "
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_0[1], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_0[1], 3))
        + "}}$ "
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_0[2], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_0[2], 3))
        + "}}$ "
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_0[3], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_0[3], 3))
        + "}}$ "
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_0[4], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_0[4], 3))
        + "}}$ "
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_0[5], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_0[5], 3))
        + "}}$ "
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.55, 0),
        fontsize=12,
        labelspacing=0.1,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs01[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks_0
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray_0[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin_0,
            vmax=vmax_0,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.055,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split_0[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    cax = fig.add_axes([0.92, 0.54, 0.01, 0.33])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="vertical",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )

    # figure part 2
    fig_lst = [
        "(b)",
        "(b1)",
        "(b2)",
        "(b3)",
        "(b4)",
        "(b5)",
        "(b6)",
        "(b7)",
    ]

    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_all[1], width_ratios=[1.1, 1.75]
    )

    gs01 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs1[0]
    )

    axbig = fig.add_subplot(gs01[0, 0])

    axbig.plot(
        data_x_1[0],
        data_y_1[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap_1[0])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[1]),
    )
    axbig.plot(
        data_x_1[1],
        data_y_1[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap_1[1])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[2]),
    )
    axbig.plot(
        data_x_1[2],
        data_y_1[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap_1[2])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[3]),
    )
    axbig.plot(
        data_x_1[3],
        data_y_1[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap_1[3])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[4]),
    )
    axbig.plot(
        data_x_1[4],
        data_y_1[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap_1[4])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[5]),
    )
    axbig.plot(
        data_x_1[5],
        data_y_1[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap_1[5])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin_1, ymax_1)
    axbig.set_xlim(xmin_1, xmax_1)
    axbig.set_yticks(yticks_1)

    axbig.legend(
        loc="lower left",
        handlelength=4.5,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.53,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_1[0], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_1[0], 3))
        + "}}$ "
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_1[1], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_1[1], 3))
        + "}}$ "
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_1[2], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_1[2], 3))
        + "}}$ "
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_1[3], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_1[3], 3))
        + "}}$ "
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_1[4], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_1[4], 3))
        + "}}$ "
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(intercept_1[5], 1))
        + " * "
        + x_var_name
        + "$\mathregular{^{"
        + str(np.round(slope_1[5], 3))
        + "}}$ "
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.55, 0),
        fontsize=12,
        labelspacing=0.1,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs02 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs1[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs02[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs02[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs02[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs02[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks_1
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray_1[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin_1,
            vmax=vmax_1,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.055,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split_1[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    plt.subplots_adjust(
        wspace=0.13, hspace=0.15,
    )

    cax = fig.add_axes([0.92, 0.132, 0.01, 0.33])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="vertical",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )
    plt.savefig(
        "figure/"
        + data_x_label
        + "_"
        + data_y_label
        + "_"
        + "_double.png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )


def plot_both_nolog(
    # line plot part
    data_x,
    data_y,
    main_gap,
    slope,
    intercept,
    yticks,
    data_x_label,
    data_y_label,
    main_gap_name,
    x_var_name,
    y_var_name,
    ymin,
    ymax,
    xmin,
    xmax,
    linewidth,
    # spatial plot part
    dataarray,
    p_spatial_split,
    min,
    max,
    var_name,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot both line plot and spatial plot.

    Args:
        data_x (np.array): x data for line plot
        data_y (np.array): y data for line plot
        main_gap (np.array): main gap for line plot
        slope (np.array): slope for linear regression
        intercept (np.array): intercept for linear regression
        data_x_label (str): x label for line plot
        data_y_label (str): y label for line plot
        ymin (float): min value of y axis for line plot
        ymax (float): max value of y axis for line plot
        xmin (float): min value of x axis for line plot
        xmax (float): max value of x axis for line plot
        linewidth (float): linewidth for line plot
        dataarray (np.array): data for spatial plot slope
        p_spatial_split (np.array): p value for spatial plot
        min (float): min value for colorbar
        max (float): max value for colorbar
        var_name (str): variable name for colorbar
        cmap_file (str, optional): file dir for cmap. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.gridspec as gridspec

    fig_lst = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
    ]

    fig = plt.figure(figsize=(18.5, 5), dpi=300)

    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.8])

    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs0[0]
    )

    axbig = fig.add_subplot(gs00[0, 0])

    axbig.plot(
        data_x[0],
        data_y[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap[0])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[1]),
    )
    axbig.plot(
        data_x[1],
        data_y[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap[1])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[2]),
    )
    axbig.plot(
        data_x[2],
        data_y[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap[2])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[3]),
    )
    axbig.plot(
        data_x[3],
        data_y[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap[3])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[4]),
    )
    axbig.plot(
        data_x[4],
        data_y[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap[4])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[5]),
    )
    axbig.plot(
        data_x[5],
        data_y[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap[5])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin, ymax)
    axbig.set_xlim(xmin, xmax)
    axbig.set_yticks(yticks)

    axbig.legend(
        loc="lower left",
        handlelength=4.5,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.568,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[0], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[0], 1))
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[1], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[1], 1))
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[2], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[2], 1))
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[3], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[3], 1))
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[4], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[4], 1))
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[5], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[5], 1))
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.52, 0),
        fontsize=11,
        labelspacing=0.75,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs01[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min,
            vmax=max,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.88,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    plt.subplots_adjust(
        wspace=0.13, hspace=0.15,
    )

    cax = fig.add_axes([0.45, -0.02, 0.43, 0.03])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="horizontal",
        # location="bottom",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )
    plt.savefig(
        "figure/"
        + data_x_label
        + "_"
        + data_y_label
        + "_"
        + main_gap_name
        + ".png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )


def plot_both_nolog_pc1_iwp(
    # figure part1
    # line plot part
    data_x_0,
    data_y_0,
    main_gap_0,
    slope_0,
    intercept_0,
    yticks_0,
    main_gap_name_0,
    ymin_0,
    ymax_0,
    xmin_0,
    xmax_0,
    # spatial plot part
    dataarray_0,
    p_spatial_split_0,
    vmin_0,
    vmax_0,
    # figure part2
    # line plot part
    data_x_1,
    data_y_1,
    main_gap_1,
    slope_1,
    intercept_1,
    yticks_1,
    main_gap_name_1,
    ymin_1,
    ymax_1,
    xmin_1,
    xmax_1,
    # spatial plot part
    dataarray_1,
    p_spatial_split_1,
    vmin_1,
    vmax_1,
    # Universal settings
    data_x_label,
    data_y_label,
    x_var_name,
    y_var_name,
    linewidth,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot both line plot and spatial plot.

    Args:
        data_x (np.array): x data for line plot
        data_y (np.array): y data for line plot
        main_gap (np.array): main gap for line plot
        slope (np.array): slope for linear regression
        intercept (np.array): intercept for linear regression
        data_x_label (str): x label for line plot
        data_y_label (str): y label for line plot
        ymin (float): min value of y axis for line plot
        ymax (float): max value of y axis for line plot
        xmin (float): min value of x axis for line plot
        xmax (float): max value of x axis for line plot
        linewidth (float): linewidth for line plot
        dataarray (np.array): data for spatial plot slope
        p_spatial_split (np.array): p value for spatial plot
        min (float): min value for colorbar
        max (float): max value for colorbar
        var_name (str): variable name for colorbar
        cmap_file (str, optional): file dir for cmap. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.gridspec as gridspec

    fig_lst = [
        "(a)",
        "(a1)",
        "(a2)",
        "(a3)",
        "(a4)",
        "(a5)",
        "(a6)",
        "(a7)",
    ]

    fig = plt.figure(figsize=(18, 10.15), dpi=300)

    gs_all = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, 1], hspace=0.2
    )

    # figure part1
    gs0 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_all[0], width_ratios=[1.1, 1.75]
    )

    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs0[0]
    )

    axbig = fig.add_subplot(gs00[0, 0])

    axbig.plot(
        data_x_0[0],
        data_y_0[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap_0[0])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[1]),
    )
    axbig.plot(
        data_x_0[1],
        data_y_0[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap_0[1])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[2]),
    )
    axbig.plot(
        data_x_0[2],
        data_y_0[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap_0[2])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[3]),
    )
    axbig.plot(
        data_x_0[3],
        data_y_0[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap_0[3])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[4]),
    )
    axbig.plot(
        data_x_0[4],
        data_y_0[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap_0[4])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[5]),
    )
    axbig.plot(
        data_x_0[5],
        data_y_0[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap_0[5])
        + "<"
        + main_gap_name_0
        + "<"
        + str(main_gap_0[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin_0, ymax_0)
    axbig.set_xlim(xmin_0, xmax_0)
    axbig.set_yticks(yticks_0)

    axbig.legend(
        loc="lower left",
        handlelength=4.5,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.568,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_0[0], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_0[0], 1))
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_0[1], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_0[1], 1))
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_0[2], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_0[2], 1))
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_0[3], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_0[3], 1))
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_0[4], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_0[4], 1))
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_0[5], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_0[5], 1))
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.52, 0),
        fontsize=11,
        labelspacing=0.75,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs01[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks_0
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray_0[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin_0,
            vmax=vmax_0,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.055,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split_0[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    cax = fig.add_axes([0.92, 0.54, 0.01, 0.33])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="vertical",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )

    # figure part 2
    fig_lst = [
        "(b)",
        "(b1)",
        "(b2)",
        "(b3)",
        "(b4)",
        "(b5)",
        "(b6)",
        "(b7)",
    ]

    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_all[1], width_ratios=[1.1, 1.75]
    )

    gs01 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs1[0]
    )

    axbig = fig.add_subplot(gs01[0, 0])

    axbig.plot(
        data_x_1[0],
        data_y_1[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap_1[0])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[1]),
    )
    axbig.plot(
        data_x_1[1],
        data_y_1[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap_1[1])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[2]),
    )
    axbig.plot(
        data_x_1[2],
        data_y_1[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap_1[2])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[3]),
    )
    axbig.plot(
        data_x_1[3],
        data_y_1[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap_1[3])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[4]),
    )
    axbig.plot(
        data_x_1[4],
        data_y_1[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap_1[4])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[5]),
    )
    axbig.plot(
        data_x_1[5],
        data_y_1[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap_1[5])
        + "<"
        + main_gap_name_1
        + "<"
        + str(main_gap_1[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin_1, ymax_1)
    axbig.set_xlim(xmin_1, xmax_1)
    axbig.set_yticks(yticks_1)

    axbig.legend(
        loc="lower left",
        handlelength=4.5,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.568,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_1[0], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_1[0], 1))
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_1[1], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_1[1], 1))
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_1[2], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_1[2], 1))
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_1[3], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_1[3], 1))
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_1[4], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_1[4], 1))
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope_1[5], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept_1[5], 1))
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.52, 0),
        fontsize=11,
        labelspacing=0.75,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs02 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs1[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs02[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs02[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs02[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs02[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks_1
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray_1[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin_1,
            vmax=vmax_1,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.055,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split_1[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    plt.subplots_adjust(
        wspace=0.13, hspace=0.15,
    )

    cax = fig.add_axes([0.92, 0.132, 0.01, 0.33])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="vertical",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )
    plt.savefig(
        "figure/"
        + data_x_label
        + "_"
        + data_y_label
        + "_"
        + "_double.png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )


def plot_both_nolog_swlw(
    # line plot part
    data_x,
    data_y,
    main_gap,
    slope,
    intercept,
    yticks,
    data_x_label,
    data_y_label,
    main_gap_name,
    x_var_name,
    y_var_name,
    ymin,
    ymax,
    xmin,
    xmax,
    linewidth,
    # spatial plot part
    dataarray,
    p_spatial_split,
    min,
    max,
    var_name,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot both line plot and spatial plot.

    Args:
        data_x (np.array): x data for line plot
        data_y (np.array): y data for line plot
        main_gap (np.array): main gap for line plot
        slope (np.array): slope for linear regression
        intercept (np.array): intercept for linear regression
        data_x_label (str): x label for line plot
        data_y_label (str): y label for line plot
        ymin (float): min value of y axis for line plot
        ymax (float): max value of y axis for line plot
        xmin (float): min value of x axis for line plot
        xmax (float): max value of x axis for line plot
        linewidth (float): linewidth for line plot
        dataarray (np.array): data for spatial plot slope
        p_spatial_split (np.array): p value for spatial plot
        min (float): min value for colorbar
        max (float): max value for colorbar
        var_name (str): variable name for colorbar
        cmap_file (str, optional): file dir for cmap. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.gridspec as gridspec

    fig_lst = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
    ]

    fig = plt.figure(figsize=(19.5, 5), dpi=300)

    gs0 = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.15, 1.9]
    )

    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs0[0]
    )

    axbig = fig.add_subplot(gs00[0, 0])

    axbig.plot(
        data_x[0],
        data_y[0],
        linestyle="solid",
        marker="^",
        linewidth=linewidth,
        color="cornflowerblue",
        label=str(main_gap[0])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[1]),
    )
    axbig.plot(
        data_x[1],
        data_y[1],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="skyblue",
        label=str(main_gap[1])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[2]),
    )
    axbig.plot(
        data_x[2],
        data_y[2],
        linestyle="dashed",
        marker="o",
        linewidth=linewidth,
        color="orange",
        label=str(main_gap[2])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[3]),
    )
    axbig.plot(
        data_x[3],
        data_y[3],
        linestyle=(0, (3, 1, 1, 1)),
        marker="D",
        linewidth=linewidth,
        color="coral",
        label=str(main_gap[3])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[4]),
    )
    axbig.plot(
        data_x[4],
        data_y[4],
        linestyle=(0, (3, 1, 1, 1)),
        marker="v",
        linewidth=linewidth,
        color="orangered",
        label=str(main_gap[4])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[5]),
    )
    axbig.plot(
        data_x[5],
        data_y[5],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker="p",
        color="firebrick",
        linewidth=linewidth,
        label=str(main_gap[5])
        + "<"
        + main_gap_name
        + "<"
        + str(main_gap[6]),
    )
    axbig.set_xlabel(data_x_label, fontsize=14)
    axbig.set_ylabel(data_y_label, fontsize=14)
    axbig.set_ylim(ymin, ymax)
    axbig.set_xlim(xmin, xmax)
    axbig.set_yticks(yticks)

    axbig.legend(
        loc="lower left",
        handlelength=4.6,
        bbox_to_anchor=(0, 0),
        fontsize=12,
        labelspacing=0.555,
    )

    axbig1 = axbig.twinx()
    axbig1.get_yaxis().set_visible(False)

    patch0 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[0], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[0], 1))
    )
    patch1 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[1], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[1], 1))
    )
    patch2 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[2], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[2], 1))
    )
    patch3 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[3], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[3], 1))
    )
    patch4 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[4], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[4], 1))
    )
    patch5 = mpatches.Patch(
        label=" "
        + y_var_name
        + " = "
        + str(np.round(slope[5], 3))
        + " * "
        + x_var_name
        + " + "
        + str(np.round(intercept[5], 1))
    )
    axbig1.legend(
        handles=[patch0, patch1, patch2, patch3, patch4, patch5],
        loc="lower left",
        handlelength=0,
        bbox_to_anchor=(0.48, 0),
        fontsize=11,
        labelspacing=0.75,
    )
    axbig.grid(True, which="both", ls="-.", color="0.85")
    axbig.text(
        0.015,
        0.944,
        fig_lst[0],
        transform=axbig.transAxes,
        size=15,
        weight="bold",
    )

    ####################################
    # spatial distribution part ########
    ####################################

    states_shp = Reader("DBATP/DBATP_Line.shp")

    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )
    import matplotlib.ticker as mticker

    # cmap = dcmap(cmap_file)
    # cmap.set_bad("gray", alpha=0)
    # cmap.set_over("#800000")
    # cmap.set_under("#191970")
    cmap = "RdBu_r"

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[1], hspace=0.09, wspace=0.04
    )

    axs_right = []
    axs_right.append(
        fig.add_subplot(gs01[0, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[0, 1], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 0], projection=ccrs.PlateCarree())
    )
    axs_right.append(
        fig.add_subplot(gs01[1, 1], projection=ccrs.PlateCarree())
    )
    # axs_right = axs[:, 1:].flatten()

    for i in range(4):

        # set the map extent
        axs_right[i].set_extent(
            [70, 105, 25, 40], crs=ccrs.PlateCarree()
        )
        # axs[i].set_facecolor("silver")

        # set xticks and yticks
        if i == 0 or i == 2:
            axs_right[i].set_yticks(
                [26, 30, 34, 38], crs=ccrs.PlateCarree()
            )
            lat_formatter = LatitudeFormatter()
            axs_right[i].yaxis.set_major_formatter(lat_formatter)

        if i == 2 or i == 3:
            axs_right[i].set_xticks(
                [75, 80, 85, 90, 95, 100], crs=ccrs.PlateCarree()
            )
            lon_formatter = LongitudeFormatter()
            axs_right[i].xaxis.set_major_formatter(lon_formatter)

        # add the shapefile of Tibet
        axs_right[i].add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        # draw gridlines
        gl = axs_right[i].gridlines(
            ccrs.PlateCarree(),
            linestyle="-.",
            lw=0.2,
            alpha=0.5,
            draw_labels=False,
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False

        # main pcolormesh plot
        a = axs_right[i].pcolormesh(
            lon,
            lat,
            dataarray[i, :, :],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min,
            vmax=max,
        )

        # add (a), (b), (c) to the plot
        axs_right[i].text(
            0.015,
            0.88,
            fig_lst[i + 1],
            transform=axs_right[i].transAxes,
            size=15,
            weight="bold",
        )

        # dotted area
        lons, lats = np.meshgrid(lon, lat)

        dot_area = np.where(p_spatial_split[i] < 0.05)
        dot = axs_right[i].scatter(
            lons[dot_area],
            lats[dot_area],
            color="k",
            s=3,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )

    plt.subplots_adjust(
        wspace=0.13, hspace=0.15,
    )

    cax = fig.add_axes([0.45, -0.02, 0.43, 0.03])
    cb = plt.colorbar(
        a,
        ax=axs_right,
        orientation="horizontal",
        # location="bottom",
        shrink=0.8,
        extend="both",
        # pad=0.4,
        aspect=45,
        cax=cax,
    )
    plt.savefig(
        "figure/"
        + data_x_label
        + "_"
        + data_y_label
        + "_"
        + main_gap_name
        + ".png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )


def test_spatial(dataset):
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(dataset)
    plt.show()


def plot_spatial_distribution_of_var(
    dataarray, min, max, var_name, cmap_file="Color/PC1_color.txt",
):
    """
    Plot spatial distribution of dataarray with spatial split.

    Args:
        dataarray (np.array): main dataarray of slope
        p_spatial_split (np.array): main dataarray of p value
        min (float): min value of colorbar
        max (float): max value of colorbar
        var_name (str): name of variable
        cmap_file (str, optional): cmap file directory. Defaults to "Color/PC1_color.txt".
    """
    # read in shapefile
    states_shp = Reader("DBATP/DBATP_Line.shp")

    # set up lat lon
    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("white")

    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(12, 7.5),
        # tight_layout=True,
        constrained_layout=True,
    )
    axs = plt.subplot(111, projection=ccrs.PlateCarree(),)
    # plt.rcParams.update({"font.family": "Times New Roman"})

    axs.set_extent([70, 105, 25, 40], crs=ccrs.PlateCarree())
    # axs[i].set_facecolor("silver")

    a = axs.pcolormesh(
        lon,
        lat,
        dataarray,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=min,
        vmax=max,
    )

    axs.add_geometries(
        states_shp.geometries(),
        ccrs.PlateCarree(),
        linewidths=0.7,
        facecolor="none",
        edgecolor="black",
    )

    gl = axs.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    gl.left_labels = False

    cb = plt.colorbar(
        a,
        ax=axs,
        location="bottom",
        shrink=0.93,
        extend="both",
        pad=0.08,
        aspect=45,
        label=var_name,
    )

    plt.savefig(
        var_name + ".png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )
    # cb.set_label(label=var_name, size=24)

    # cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    # gl.xlabel_style = {"size": 18}
    # gl.ylabel_style = {"size": 18}


def plot_spatial_distribution_of_6var_fig_og(
    var_dataarray,
    # min,
    # max,
    var_name,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot spatial distribution of dataarray with spatial split.

    Args:
        dataarray (np.array): main dataarray of slope
        p_spatial_split (np.array): main dataarray of p value
        min (float): min value of colorbar
        max (float): max value of colorbar
        var_name (str): name of variable
        cmap_file (str, optional): cmap file directory. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.ticker as mticker

    # read in shapefile
    states_shp = Reader("DBATP/DBATP_Line.shp")

    # set up lat lon
    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("white")

    fig, axs = plt.subplots(
        nrows=6,
        ncols=1,
        figsize=(6, 11),
        # tight_layout=True,
        constrained_layout=True,
    )

    for var in range(6):
        axs = plt.subplot(
            6, 1, var + 1, projection=ccrs.PlateCarree(),
        )

        axs.set_extent([70, 105, 25, 40], crs=ccrs.PlateCarree())

        a = axs.pcolormesh(
            lon,
            lat,
            var_dataarray[var],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            # vmin=min,
            # vmax=max,
        )

        axs.add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        gl = axs.gridlines(
            linestyle="-.", lw=0.8, alpha=0.8, draw_labels=True
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        if var == 5:
            gl.bottom_labels = True

        cb = plt.colorbar(
            a,
            ax=axs,
            location="right",
            shrink=0.9,
            extend="both",
            # pad=0.08,
            # aspect=45,
            label=var_name[var],
        )

    plt.savefig(
        "figure/6var_spatial.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )
    # cb.set_label(label=var_name, size=24)

    # cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    # gl.xlabel_style = {"size": 18}
    # gl.ylabel_style = {"size": 18}


def plot_spatial_distribution_of_6var_fig(
    var_dataarray,
    min,
    max,
    var_name,
    var_cbar_ticks,
    cmap_file="Color/PC1_color.txt",
):
    """
    Plot spatial distribution of dataarray with spatial split.

    Args:
        dataarray (np.array): main dataarray of slope
        p_spatial_split (np.array): main dataarray of p value
        min (float): min value of colorbar
        max (float): max value of colorbar
        var_name (str): name of variable
        cmap_file (str, optional): cmap file directory. Defaults to "Color/PC1_color.txt".
    """
    import matplotlib.ticker as mticker

    # read in shapefile
    states_shp = Reader("DBATP/DBATP_Line.shp")

    # set up lat lon
    lon = np.linspace(70, 105, 37)
    lat = np.linspace(25, 40, 17)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("white")

    fig, axs = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 6.5),
        constrained_layout=True,
    )

    for var in range(6):
        axs = plt.subplot(
            3, 2, var + 1, projection=ccrs.PlateCarree(),
        )

        axs.set_extent([70, 105, 25, 40], crs=ccrs.PlateCarree())

        a = axs.pcolormesh(
            lon,
            lat,
            var_dataarray[var],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min[var],
            vmax=max[var],
        )

        axs.add_geometries(
            states_shp.geometries(),
            ccrs.PlateCarree(),
            linewidths=0.7,
            facecolor="none",
            edgecolor="black",
        )

        gl = axs.gridlines(
            linestyle="-.", lw=0.8, alpha=0.8, draw_labels=True
        )
        gl.ylocator = mticker.FixedLocator([26, 30, 34, 38])
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        if var == 4 or var == 5:
            gl.bottom_labels = True

        cb = plt.colorbar(
            a,
            ax=axs,
            location="right",
            shrink=0.85,
            extend="both",
            # pad=0.08,
            # aspect=45,
            label=var_name[var],
            ticks=var_cbar_ticks[var],
        )

    plt.savefig(
        "figure/6var_spatial.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )
    # cb.set_label(label=var_name, size=24)

    # cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    # gl.xlabel_style = {"size": 18}
    # gl.ylabel_style = {"size": 18}


if __name__ == "__main__":
    pass
