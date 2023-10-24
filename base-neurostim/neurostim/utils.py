#%%
from neuron import h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import copy
from collections import OrderedDict
from ast import literal_eval
import re

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

def str2OrderedDict(string):
    # Remove ordered dict syntax from string by indexing
    string=string[13:]
    string=string[:-2]

    # convert string to list
    file_list=literal_eval(string)

    header=OrderedDict()
    for entry in file_list:
        # Extract key and value from each tuple
        key, value=entry
        # Create entry in OrderedDict
        header[key]=value
    return header

def str_to_lvl2nestedOrderedDict(string):
    
    OD = []
    while "OrderedDict" in string:
        idx_start = [m.start() for m in re.finditer('OrderedDict', string)][::-1][0]
        idx_end = [m.start() for m in re.finditer('\)]\)', string)][0]
        OD.append(str2OrderedDict(string[idx_start:idx_end+3]))
        string = string[:idx_start]+ "'PLACEHOLDER'" + string[idx_end+3:]
    for i in range(len(OD)-1):
        outerOD = copy.copy(OD[i+1])
        innerOD = copy.copy(OD[i])
        pos_innerOD = [i for i,x in enumerate(outerOD.values()) if x == 'PLACEHOLDER'][0]
        outerOD[list(outerOD.keys())[pos_innerOD]] = innerOD
        OD[i+1] = outerOD
    return OD[-1]

def unique_param_str_divpaths(keys_in_order, params=None):
    if params == None:
        # return str with wildcards
        return "/".join(["-".join([key, "{" + key + "}"]) for key in keys_in_order])
    return "/".join(["-".join([key, str(params[key])]) for key in keys_in_order])

def unique_param_str(keys_in_order, params=None):
    """Get a string encoding workflow params in specific order"""
    if params == None:
        # return str with wildcards
        return "--".join(["-".join([key, "{" + key + "}"]) for key in keys_in_order])
    return "--".join(["-".join([key, str(params[key])]) for key in keys_in_order])

def interpolate(df, interpolation_dt):
    time = np.arange(df["time [ms]"].min(), df["time [ms]"].max(), interpolation_dt)
    df_int = pd.DataFrame({})
    for name, values in df.items():
        f = interp1d(df["time [ms]"], values)
        df_int[name] = f(time)
    return df_int

def rm_mech(mech, sec):
    """
    Utility to remove individual mechs from cell.
    """
    mt = h.MechanismType(0)
    mt.select(mech)
    mt.remove(sec=sec)
