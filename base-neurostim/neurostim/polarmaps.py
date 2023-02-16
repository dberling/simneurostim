import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def _get_radius_angle_amp(file):
    amp = float(re.search("--amp-(.*?)--delay", file).group(1))
    angle = float(re.search("--angle-(.*?).pickle", file).group(1))
    radius = float(re.search("--radius-(.*?)--angle", file).group(1))
    return radius, angle, amp


def _get_tts(data, t_on):
    """Find time until first spike(V_soma > 0)"""
    return data["time [ms]"].loc[data["V_soma(0.5)"] > 0].values[0] - t_on


def _get_intVsoma(data, t_on, t_off, interpol_dt):
    """Calculate integral of soma voltage between t_on and t_off"""
    v_init = -71
    cond = (data["time [ms]"] >= t_on) & (data["time [ms]"] <= t_off)
    intVsoma = np.sum(interpol_dt * (data["V_soma(0.5)"].loc[cond] - v_init))
    return intVsoma


def generate_df(inputfiles, t_on, t_off, interpol_dt):
    all_data = []
    for file in inputfiles:
        results = dict()
        radius, angle, amp = _get_radius_angle_amp(file)
        data = pd.read_pickle(file)
        AP = np.any(data["V_soma(0.5)"] > 0)
        results["radius"] = radius
        results["angle"] = angle
        results["amp"] = amp
        results["AP"] = AP
        if AP:
            results["tts"] = _get_tts(data, t_on)
            results["int(V_soma)"] = np.nan
        else:
            results["tts"] = np.nan
            results["int(V_soma)"] = _get_intVsoma(data, t_on, t_off, interpol_dt)
        all_data.append(results)
    return pd.DataFrame(all_data)


def get_AP_times(df, interpol_dt, t_on, AP_threshold=None):
    """
    Find number of action potentials (APs)

    Check for following things:
    - AP happens after stimulation onset
    - peak finder relies on data 1ms before and after
    """
    # check 1ms before and after peak to be lower
    n_points_1ms = int(1 / interpol_dt)
    peak_times = df.loc[
        argrelextrema(
            df["V_soma(0.5)"].values, 
            np.greater_equal, 
            order=n_points_1ms)
    ]["time [ms]"]
    # eliminate all before stim on:
    peak_times = peak_times.loc[peak_times > t_on]
    # elimante all below threshold
    if AP_threshold != None:
        peak_values = np.array(
            [df.loc[df['time [ms]'] == time]['V_soma(0.5)'].values[0] for time in peak_times]
        )
        peak_times = peak_times[peak_values > AP_threshold]
    return peak_times


def polar_cmap(data, vmin_tts=None, vmax_tts=None, vmin_intV=None, vmax_intV=None):
    amplitudes = data.amp.unique()
    radii = data.radius.unique()
    angles = data.angle.unique()
    rmesh, ang_mesh = np.meshgrid(radii, angles)

    figheight = len(amplitudes)
    fig, axs = plt.subplots(
        2, figheight, subplot_kw={"projection": "polar"}, figsize=(5 * figheight, 10)
    )
    if vmin_tts == None:
        vmin_tts = data["tts"].min()
    if vmax_tts == None:
        vmax_tts = data["tts"].max()
    if vmin_intV == None:
        vmin_intV = data["int(V_soma)"].min()
    if vmax_intV == None:
        vmax_intV = data["int(V_soma)"].max()

    for i, amp in enumerate(amplitudes):
        pldata = data.loc[data["amp"] == amp]
        intVplot = axs[0, i].pcolormesh(
            ang_mesh,
            rmesh,
            pldata.pivot("angle", "radius", "int(V_soma)").values,
            cmap="Blues",
            shading="nearest",
            vmin=vmin_intV,
            vmax=vmax_intV,
        )
        # axs[0,i].plot(angles, radii, color='k', ls='none')
        axs[0, i].grid()

        ttsplot = axs[1, i].pcolormesh(
            ang_mesh,
            rmesh,
            pldata.pivot("angle", "radius", "tts").values,
            cmap="Greens",
            shading="nearest",
            vmin=vmin_tts,
            vmax=vmax_tts,
        )
        # axs[1,i].plot(angles, radii, color='k', ls='none')
        axs[1, i].grid()
        axs[0, i].set_title("amplitude: %2.6f" % amp)
    cax1 = fig.add_axes([0.93, 0.6, 0.01, 0.3])
    cax2 = fig.add_axes([0.93, 0.1, 0.01, 0.3])
    cbar_intV = plt.colorbar(intVplot, cax=cax1)
    cbar_tts = plt.colorbar(ttsplot, cax=cax2)
    cbar_intV.set_label("integrated depolarization [ms*mV]")
    cbar_tts.set_label("time to 1st spike [ms]")

    return fig, axs


def polar_cmap2(data, vmin_tts=None, vmax_tts=None, vmin_intV=None, vmax_intV=None):
    amplitudes = data.amp.unique()
    radii = data.radius.unique()
    angles = data.angle.unique()
    rmesh, ang_mesh = np.meshgrid(radii, angles)

    figheight = len(amplitudes) + 1
    fig, axs = plt.subplots(
        1, figheight, subplot_kw={"projection": "polar"}, figsize=(5 * figheight, 10)
    )
    if vmin_tts == None:
        vmin_tts = data["tts"].min()
    if vmax_tts == None:
        vmax_tts = data["tts"].max()
    if vmin_intV == None:
        vmin_intV = data["int(V_soma)"].min()
    if vmax_intV == None:
        vmax_intV = data["int(V_soma)"].max()

    for i, amp in enumerate(amplitudes):
        pldata = data.loc[data["amp"] == amp]
        intVplot = axs[i].pcolormesh(
            ang_mesh,
            rmesh,
            pldata.pivot("angle", "radius", "int(V_soma)").values,
            cmap="Blues",
            shading="nearest",
            vmin=vmin_intV,
            vmax=vmax_intV,
        )
        axs[i].grid()

        ttsplot = axs[i].pcolormesh(
            ang_mesh,
            rmesh,
            pldata.pivot("angle", "radius", "tts").values,
            cmap="Greens_r",
            shading="nearest",
            vmin=vmin_tts,
            vmax=vmax_tts,
        )
        axs[i].grid()
        axs[i].set_title("amplitude: %2.6f" % amp)
    axs[-1].axis("off")
    cax1 = fig.add_axes([0.85, 0.3, 0.01, 0.4])
    cax2 = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    cbar_intV = plt.colorbar(intVplot, cax=cax1)
    cbar_tts = plt.colorbar(ttsplot, cax=cax2)
    cbar_intV.set_label("integrated depolarization [ms*mV]")
    cbar_tts.set_label("time to 1st spike [ms]")

    return fig, axs


def simple_polar_map(
    data,
    plot_col,
    ax,
    cmap="Blues",
    **pcolormesh_kwargs
):
    if type(cmap) == str:
        cmap = matplotlib.cm.get_cmap(cmap)
    rmesh, ang_mesh = np.meshgrid(
        data['radius [um]'].unique(), data['angle [rad]'].unique()
    )
    ax.grid()
    mappable = ax.pcolormesh(
        ang_mesh,
        rmesh,
        data.pivot("angle [rad]", "radius [um]", plot_col).values,
        cmap=cmap,
        shading="nearest",
        **pcolormesh_kwargs
    )
    return ax, mappable
