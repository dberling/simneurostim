from copy import copy
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math

from scipy.constants import h, c
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

from neurostim.stimulator import MultiStimulator
from neurostim.chrimson_conductance_model import define_sampling, calc_ChR_conductances_numpy

from neat import GreensTree
from neat import FourrierTools

# finding spatially distributed locations on morphology to serve as approximation
## VARIANT 1: density clustering
def get_section_coordinates(section):
    x, y, z = [], [], []
    for i in range(section.n3d()):
        x.append(section.x3d(i))  # Get x-coordinate at position 0 (start)
        y.append(section.y3d(i))  # Get y-coordinate at position 0 (start)
        z.append(section.z3d(i))  # Get z-coordinate at position 0 (start)
    return np.mean(x), np.mean(y), np.mean(z)  # Return average coordinates

def sample_by_volume(positions, n_target):
    """
    Sample points from the morphology based on local volume approximation
    using nearest neighbors to estimate local density.

    Args:
        positions (np.ndarray): (N, 3) array of 3D coordinates representing the morphology.
        n_target (int): Desired number of sample points.

    Returns:
        np.ndarray: (n_target, 3) array of sampled 3D points.
    """
    # Create a KDTree for nearest neighbor searches
    tree = KDTree(positions)

    # Estimate local density by calculating the distance to the k-th nearest neighbor
    k = 10  # Number of nearest neighbors to consider
    distances, _ = tree.query(positions, k=k)

    # Use the distance to the 10th nearest neighbor as a proxy for local volume
    local_volume = distances[:, -1] ** 3  # Approximate local volume

    # Normalize the local volumes to get sampling probabilities
    sampling_probabilities = local_volume / np.sum(local_volume)

    # Perform weighted random sampling
    sampled_indices = np.random.choice(np.arange(len(positions)), size=n_target, p=sampling_probabilities)

    return positions[sampled_indices]

def optimize_positions(positions, n_iterations=100, learning_rate=0.001, max_force=0.1, min_distance=1e-6):
    """
    Optimize the positions to spread them out more evenly using electrostatic repulsion.

    Args:
        positions (np.ndarray): (n_target, 3) array of 3D coordinates to optimize.
        n_iterations (int): Number of optimization iterations.
        learning_rate (float): Step size for the optimization.
        max_force (float): Maximum allowable force to avoid runaway behavior.
        min_distance (float): Minimum distance to avoid division by zero.

    Returns:
        np.ndarray: Optimized positions.
    """
    for iteration in range(n_iterations):
        distances = squareform(pdist(positions))
        np.fill_diagonal(distances, np.inf)  # Avoid division by zero for self-distances

        # Prevent zero or very small distances
        distances = np.maximum(distances, min_distance)

        forces = np.zeros_like(positions)
        for i in range(len(positions)):
            direction_vectors = positions[i] - positions
            inverse_distances = 1.0 / distances[i][:, None]
            repulsion = direction_vectors * inverse_distances**3

            # Clip repulsion forces to prevent runaway values
            repulsion = np.clip(repulsion, -max_force, max_force)
            forces[i] = np.sum(repulsion, axis=0)

        # Move points according to the calculated forces
        positions += learning_rate * forces

        # Debugging: Check for non-finite values after each iteration
        if not np.all(np.isfinite(positions)):
            print(f"Iteration {iteration}: Non-finite values detected in positions")
            print(f"Positions: {positions}")
            raise ValueError("Non-finite values encountered during optimization.")

    return positions

def project_to_morphology(positions, morphology):
    """
    Project points back onto the morphology by finding the nearest point in the original set.

    Args:
        positions (np.ndarray): (n_target, 3) array of optimized 3D coordinates.
        morphology (np.ndarray): (N, 3) array of original 3D morphology points.

    Returns:
        np.ndarray: Projected points.
    """
    tree = KDTree(morphology)
    _, indices = tree.query(positions)
    return morphology[indices]

def find_sampled_points(morphology, n_target, n_iterations=100):
    """
    Find evenly distributed sample points on the morphology.

    Args:
        morphology (np.ndarray): (N, 3) array of 3D coordinates representing the morphology.
        n_target (int): Desired number of sample points.
        n_iterations (int): Number of optimization iterations.

    Returns:
        np.ndarray: (n_target, 3) array of sampled and optimized 3D points.
    """
    # Ensure morphology does not contain NaN or infinite values
    morphology = morphology[np.all(np.isfinite(morphology), axis=1)]

    if morphology.shape[0] < n_target:
        raise ValueError("Number of valid points in the morphology is less than n_target.")

    # Step 1: Sample initial points using volume-weighted sampling
    sampled_points = sample_by_volume(morphology, n_target)

    # Step 2: Optimize the sampled points for even distribution
    optimized_points = optimize_positions(sampled_points, n_iterations=n_iterations)

    # Ensure optimized points contain only finite values
    optimized_points = optimized_points[np.all(np.isfinite(optimized_points), axis=1)]

    if optimized_points.shape[0] < n_target:
        raise ValueError("After optimization, fewer points remain due to NaN or inf values.")

    # Step 3: Project the optimized points back onto the original morphology
    final_points = project_to_morphology(optimized_points, morphology)

    return final_points

# plot to validate positions:
def project_coords(coords, rm):
    if rm=='x':
        keep_idxs = [0,2]
    elif rm=='y':
        keep_idxs = [1,2]
    elif rm=='z':
        keep_idxs = [0,1]
    else:
        raise ValueError("rm must be 'x' or 'y'")
    plotx = np.array(coords)[:,keep_idxs].T[0]
    ploty = np.array(coords)[:,keep_idxs].T[1]
    return plotx, ploty

def plot_sampled_points_on_neuron(sec_coords, sampled_points, proj_axis):
    plt.plot(
        *project_coords(sec_coords, proj_axis),
        marker='.',
        ls='',
        color='blue'
    )
    plt.plot(
        *project_coords(sampled_points, proj_axis),
        marker='.',
        ls='',
        color='orange',
        markersize=10
    )
    plt.axis('equal')
    plt.show()

# cluster sections for a cell:
def cluster_sections(cell, n_target):
    # Create a mapping of coordinates to compartment names
    secs = [sec for sec in cell.sections]
    sec_coords = np.array([get_section_coordinates(sec) for sec in cell.sections])

    # Find sampled and optimized points
    sampled_points = find_sampled_points(
        morphology=sec_coords,
        n_target=n_target
    )
    # PLotting
    for proj_axis in ['x', 'y', 'z']:
        plot_sampled_points_on_neuron(
            sec_coords,
            sampled_points,
            proj_axis
        )

    # Convert sec_coords (which may contain numpy arrays) to tuples
    sec_coords_as_tuples = [tuple(coord) for coord in sec_coords]

    # Create a mapping of tuple coordinates to compartment names
    coord_to_sec = dict(zip(sec_coords_as_tuples, secs))

    # Now, find the compartment names for each point in sampled_points
    sampled_secs = [coord_to_sec[tuple(point)] for point in sampled_points if tuple(point) in coord_to_sec]
    return sampled_secs

##############################################################
##############################################################

## VARIANT 2: cartesian grid clustering
def assign_to_cartesian_grid(sections, grid_width):
    """
    Assign sections to a cartesian grid of varying widths and return the sections
    closest to each grid point.

    Parameters:
    sections (list): List of sections.
    grid_width (float): Width of the grid cells in x, y, and z dimensions.

    Returns:
    list of sections closet to grid points.
    """
    coordinates = np.array([(get_section_coordinates(sec)) for sec in sections])
    grid_widths = np.array((grid_width, grid_width, grid_width))

    # Compute grid point indices for each element
    grid_indices = np.floor(coordinates / grid_widths).astype(int)

    # Map sections to grid points
    grid_to_sections = {}
    for i, index in enumerate(grid_indices):
        index_tuple = tuple(index)
        if index_tuple not in grid_to_sections:
            grid_to_sections[index_tuple] = []
        grid_to_sections[index_tuple].append(i)

    # Determine the closest element to each grid point
    closest_sections = {}
    for grid_point, element_indices in grid_to_sections.items():
        # Convert grid point back to spatial coordinates (grid center)
        grid_center = np.array(grid_point) * grid_widths + grid_widths / 2

        # Find the closest element to the grid center
        distances = [
            np.linalg.norm(coordinates[idx] - grid_center)
            for idx in element_indices
        ]
        closest_index = element_indices[np.argmin(distances)]
        closest_sections[grid_point] = sections[closest_index]

    return list(closest_sections.values())

##############################################################
##############################################################
##############################################################
##############################################################

# clustering other locations on morphology to locations approximating morphology
# Function to get the 3D coordinates of a section's segments
def get_section_coordinates(sec):
    x, y, z = [], [], []
    for i in range(sec.n3d()):
        x.append(sec.x3d(i))  # Get x-coordinate at position 0 (start)
        y.append(sec.y3d(i))  # Get y-coordinate at position 0 (start)
        z.append(sec.z3d(i))  # Get z-coordinate at position 0 (start)
    return np.mean(x), np.mean(y), np.mean(z)  # Return average coordinates

# Function to compute Euclidean distance between two points in 3D space
def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((np.array(coord1) - np.array(coord2))**2))

# Main function to group sections by proximity
def group_sections_by_proximity(fixed_sections, other_sections):
    fixed_coords = [get_section_coordinates(sec) for sec in fixed_sections]
    groupings = {sec: [] for sec in fixed_sections}  # Dictionary to store groupings
    grouped_distances = []
    for sec in other_sections:
        sec_coord = get_section_coordinates(sec)
        # Find the closest fixed section
        distances = [euclidean_distance(sec_coord, fixed_coord) for fixed_coord in fixed_coords]
        closest_section = fixed_sections[np.argmin(distances)]
        groupings[closest_section].append(sec)  # Assign the section to the closest fixed section
        grouped_distances.append(distances[np.argmin(distances)])

    return groupings, grouped_distances

def transform_dict(input_dict):
    """
    Transforms a dictionary where keys and list elements have a `.name()` attribute.
    The new dictionary will use the `.name()` of the keys and elements.

    Parameters:
    input_dict (dict): Dictionary with keys and lists of objects having `.name()`.

    Returns:
    dict: Transformed dictionary with `.name()` attributes as keys and values.
    """
    # Use a dictionary comprehension for transformation
    return {key.name(): [element.name() for element in value] for key, value in input_dict.items()}


##############################################################
##############################################################
##############################################################
##############################################################




# obtain cell information for conductance calculation, including impedances
def get_cell_data(cell):

    secs = cell.sections
    allsegment_locs = []
    allsegment_coords_area = []
    for sec in secs:
        for seg in sec:
            allsegment_locs.append(
                dict(
                    node=int(sec.name()),
                    x=seg.x
                )
            )
            allsegment_coords_area.append([
                seg.x_chanrhod,
                seg.y_chanrhod,
                seg.z_chanrhod,
                seg.area(),
                seg.channel_density_chanrhod
            ])
    # set up greenstree for impedance calculations
    greens_tree = cell.ph_tree.__copy__(new_tree=GreensTree())

    # calc impedance kernels from 0 to 50 ms
    # create a Fourriertools instance with the temporal array on which to evaluate the impedance kernel
    t_arr = np.linspace(0.,200,4000)
    ft = FourrierTools(t_arr)
    # appropriate frequencies are stored in `ft.s`
    # set the boundary condition for cylindrical segments in `greens_tree`
    greens_tree.setImpedance(ft.s)# calc impedance kernels from 0 to 50 ms
    # create a Fourriertools instance with the temporal array on which to evaluate the impedance kernel
    t_arr = np.linspace(0.,200,4000)
    ft = FourrierTools(t_arr)
    # appropriate frequencies are stored in `ft.s`
    # set the boundary condition for cylindrical segments in `greens_tree`
    greens_tree.setImpedance(ft.s)

    # record input resistances
    data = []
    soma_loc = allsegment_locs[0]
    for loc,coords in zip(allsegment_locs, allsegment_coords_area):
        # input resistance:
        ir = greens_tree.calcZF(loc, loc)[ft.ind_0s].real
        tr = greens_tree.calcZF(loc, soma_loc)[ft.ind_0s].real
        data.append([loc['node'], loc['x'], ir, tr, *coords])

    return np.array(data)




##############################################################
##############################################################
##############################################################
##############################################################


# calculate conductance
def calc_fluxes_photons_PER_cm2_fs(comp_xyz, stimulator, norm_power_mW_of_MultiStimulator):
    """
    Input:
    ------
    comp_xyz: list
        [x, y, z] with x,y,z each a list of coords in [um] of the sections
    stimulator: object from neurostim.stimulator
    norm_power_mW_of_MultiStimulator: float
        normalized stim_power in mW

    Returns:
    --------
    fluxes in [photons / (cm**2 * fs)]
    """
    # calculate photon flux at light source output
    E_photon = h * c / (595e-9)
    photon_flux_source_PER_s = norm_power_mW_of_MultiStimulator * 1e-3 / E_photon # Power [J/s] / Photon_energy [J]
    photon_flux_source_PER_fs = photon_flux_source_PER_s * 1e-15

    fluxes_photons_PER_cm2_fs = [
        # flux at light source output [1/fs]
        photon_flux_source_PER_fs *\
        # returns combined light transmission in 1/cm2
        stimulator.calculate_Tx_at_pos(
            pos_xyz_um =  [x_,y_,z_],
            stim_xyz_um = [0,0,0]
        ) for x_,y_,z_ in zip(*comp_xyz)
    ]
    return fluxes_photons_PER_cm2_fs

def construct_stim_times(temp_protocol, interpol_dt_ms):
    """
    Craft array of stim_times at time resolution interpol_dt_ms and
    according to delay, duration, and total_rec_time in temp_protocol.
    """
    stimulation_times = np.ones(int(temp_protocol['total_rec_time_ms']/interpol_dt_ms))
    stimulation_times[:int(temp_protocol['delay_ms']/interpol_dt_ms)] = 0
    stimulation_times[int(temp_protocol['duration_ms']/interpol_dt_ms):] = 0
    return stimulation_times

def calc_channel_cond_for_secs_and_flux_over_time(
    comp_xyz,
    stimulator,
    norm_power_mW_of_MultiStimulator,
    temp_protocol, reject_if_sampling_smaller
):
    # calculate light fluxes at compartments:
    fluxes_photons_PER_cm2_fs = calc_fluxes_photons_PER_cm2_fs(
        comp_xyz=comp_xyz,
        stimulator=stimulator,
        norm_power_mW_of_MultiStimulator=norm_power_mW_of_MultiStimulator
    )
    # generate temporal evolution of stimulation
    interpol_dt_ms = define_sampling(fluxes_photons_PER_cm2_fs)
    if interpol_dt_ms < reject_if_sampling_smaller:
        # sampling period would be to small for the calculations to be computationally feasible
        return None, interpol_dt_ms, False
    # construct stimulation times array
    stimulation_times = construct_stim_times(temp_protocol, interpol_dt_ms)
    # extend fluxes according to stim_times into time_dimension
    fluxes_photons_PER_cm2_fs = [stimulation_times * flux for flux in fluxes_photons_PER_cm2_fs]
    # calculate conductances
    channel_conductance_nS = calc_ChR_conductances_numpy(
        flux_photonsPERcm2_fs=np.array(fluxes_photons_PER_cm2_fs).T,
        sampling_period=interpol_dt_ms
    )
    return channel_conductance_nS, interpol_dt_ms, True

def get_inverse_dict(d):
    # Inverse dictionary
    inverse_dict = {}

    # Iterate over the original dictionary
    for key, values in d.items():
        for value in values:
            inverse_dict[value] = key
    return inverse_dict

def convert_dict_to_floats(d):
    float_dict = dict()
    for key in d.keys():
        float_dict[float(key)] = float(d[key])
    return float_dict

def calc_grouped_channel_cond(
    grouping, secnames, comp_xyz, stimulator, norm_power_mW_of_MultiStimulator,
    temp_protocol, reject_if_sampling_smaller
):
    grouping_secs = [secname for secname in list(grouping.keys())]
    grouping_sec_idxs = [list(secnames).index(grouping_sec) for grouping_sec in grouping_secs]
    grouping_sec_xyzs = [
        [comp_xyz[0][idx] for idx in grouping_sec_idxs],
        [comp_xyz[1][idx] for idx in grouping_sec_idxs],
        [comp_xyz[2][idx] for idx in grouping_sec_idxs]
    ]

    channel_conductance_nS, interpol_dt_ms, success = calc_channel_cond_for_secs_and_flux_over_time(
        comp_xyz=grouping_sec_xyzs,
        stimulator=stimulator,
        norm_power_mW_of_MultiStimulator=norm_power_mW_of_MultiStimulator,
        temp_protocol=temp_protocol,
        reject_if_sampling_smaller=reject_if_sampling_smaller
    )
    # return channel_conductance_nS
    # inverse grouping:
    for key in grouping.keys():
        grouping[key].append(key)
    #grouping_inv = convert_dict_to_floats(get_inverse_dict(grouping))
    grouping_inv = get_inverse_dict(grouping)

    expanded_channel_conductance_nS = []
    for secname in secnames:
        if secname not in grouping_secs:
            idx_in_grouping_secs = list(grouping_secs).index(grouping_inv[secname])
            expanded_channel_conductance_nS.append(channel_conductance_nS.T[idx_in_grouping_secs])
        else:
            idx_in_grouping_secs = list(
                grouping_secs
            ).index(secname)
            expanded_channel_conductance_nS.append(
                channel_conductance_nS.T[idx_in_grouping_secs]
            )

    return np.array(expanded_channel_conductance_nS).T, interpol_dt_ms, success

def rescale_conductance(
    channel_conductance_nS,
    area_um2,
    channel_density_PERcm2,
    input_resistance_GOhm,
    transfer_resistance_GOhm
):
    N_channel = area_um2 * 1e-8 * channel_density_PERcm2
    comp_conductance_nS = channel_conductance_nS * np.array(N_channel)
    resistance_diff = np.abs(
        input_resistance_GOhm - transfer_resistance_GOhm
    )
    rescaled_cond_nS = comp_conductance_nS / (
        1 + resistance_diff * comp_conductance_nS
    )
    return rescaled_cond_nS

def calc_grouped_rescaled_comp_conductance_nS(
    grouping,
    norm_power_mW_of_MultiStimulator,
    stimulator_config,
    comp_data,
    temp_protocol,
    reject_if_sampling_smaller=0.001
):
    # convert str(list) to real list if needed:
    if type(stimulator_config[0]['position']) == str:
        for config in stimulator_config:
            config['position'] = ast.literal_eval(config['position'])

    # load neuron comp data
    secnames, sec_x, input_resistance_MOhm, transfer_resistance_MOhm, x, y, z, area_um2, channel_density_PERcm2 = comp_data.T
    input_resistance_GOhm = input_resistance_MOhm * 1e-3
    transfer_resistance_GOhm = transfer_resistance_MOhm * 1e-3
    # convert secnames from int to str
    secnames=[str(int(sec)) for sec in secnames]

    channel_conductance_nS, interpol_dt_ms, success = calc_grouped_channel_cond(
        grouping=grouping,
        secnames=secnames,
        comp_xyz=[x,y,z],
        stimulator=MultiStimulator(stimulator_config),
        norm_power_mW_of_MultiStimulator=norm_power_mW_of_MultiStimulator,
        temp_protocol=temp_protocol,
        reject_if_sampling_smaller=reject_if_sampling_smaller
    )
    if success == False:
        return None, interpol_dt_ms, success
    else:
        rescaled_cond_nS = rescale_conductance(
            channel_conductance_nS,
            area_um2,
            channel_density_PERcm2,
            input_resistance_GOhm,
            transfer_resistance_GOhm
        )
        return rescaled_cond_nS, interpol_dt_ms, success
