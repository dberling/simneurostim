#%%
from neuron import h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


def find_mean_section_coordinates(sec):
    """Determine average coordinate for a given section"""
    n = h.n3d(sec=sec).__init__()


def convert_polar_to_cartesian_xz(radius, angle):
    x = np.cos(angle) * radius
    z = np.sin(angle) * radius
    return x, z


def plot_components(comp_dict, output):
    keys = list(comp_dict.keys())
    for k in keys:
        if k.startswith("i_") or k.startswith("I_"):
            plt.plot(comp_dict["time [ms]"], comp_dict[k], label=k)
    plt.legend()
    plt.savefig(output)


def arbitrary_3d_rotation_along_axis(point, axis, angle):
    if axis == "x":
        axis = 0
    if axis == "y":
        axis = 1
    if axis == "z":
        axis = 2
    rotvec = [0] * 3
    rotvec[axis] = angle
    r = R.from_rotvec(rotvec).as_matrix()
    rotated_point = np.dot(r, point)
    return rotated_point


def unique_param_str(keys_in_order, params=None):
    """Get a string encoding workflow params in specific order"""
    if params == None:
        # return str with wildcards
        return "--".join(["-".join([key, "{" + key + "}"]) for key in keys_in_order])
    return "--".join(["-".join([key, str(params[key])]) for key in keys_in_order])


def interpolate(df, interpolation_dt):
    time = np.arange(df["time [ms]"].min(), df["time [ms]"].max(), interpolation_dt)
    df_int = pd.DataFrame({})
    for name, values in df.iteritems():
        f = interp1d(df["time [ms]"], values)
        df_int[name] = f(time)
    return df_int
