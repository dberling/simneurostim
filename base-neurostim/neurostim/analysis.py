import numpy as np
import pandas as pd
from neuron import h
from neurostim.cell import Cell
from neurostim.stimulator import Stimulator
from neurostim.simulation import SimControl
from neurostim.utils import convert_polar_to_cartesian_xz
from scipy.signal import argrelextrema

def get_AP_times(df, interpol_dt, t_on, AP_threshold=None, apply_to="V_soma(0.5)"):
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
            df[apply_to].values, 
            np.greater_equal, 
            order=n_points_1ms)
    ]["time [ms]"]
    # eliminate all before stim on:
    peak_times = peak_times.loc[peak_times > t_on]
    # elimante all below threshold
    if AP_threshold != None:
        peak_values = np.array(
            [df.loc[df['time [ms]'] == time][apply_to].values[0] for time in peak_times]
        )
        peak_times = peak_times[peak_values > AP_threshold]
    return peak_times

def get_AP_count(df, interpol_dt_ms, t_on_ms, AP_threshold_mV, apply_to="V_soma(0.5)"):
    count = len(
        get_AP_times(
            df, interpol_dt_ms, t_on_ms, AP_threshold_mV, apply_to
        )
    )
    return count

def analyze_AP_count(sim_data, segs):
    return get_AP_count(
        sim_data,
        interpol_dt_ms=0.1,
        t_on_ms=1,
        AP_threshold_mV=0
    )

def multiply_seg_with_area(sim_data, segs):
    """
    Convert density variable (1/cm2) named str(seg) to absolute variable.
    """
    # convert density variable to absolute variable
    for seg in segs:
        # eval area of segment and convert from um2 to cm2
        sim_data[str(seg)] *= eval('h.'+str(seg)+'.area()') * 1e-8
    return sim_data

def simulate_spatial_profile(
    cell_dict,
    stimulator_dict,
    stim_intensity_mWPERmm2,
    radii_um,
    angles_rad,
    temp_protocol,
    seg_rec_vars,
    allseg_rec_var,
    sim_data_transform,
    scalar_result_names,
    scalar_result_funcs,
    vector_result_func,
    interpol_dt_ms,
    AP_threshold_mV,
    ):
    """
    Simulate a spatial neural response profile.
    
    Params:
    -------
    cell_dict: dict;
        Cell properties including:
        cellname: str;
        cortical_depth: dict;
        ChR_soma_density: float;
        ChR_distribution: str;
    stimulator_dict: dict;
        Stimulator properties:
        diameter_um: float;
        NA: float;
    stim_intensity_mWPERmm2: float;
        Stimulation intensity at stimulator output surface.
    radii_um: list of int;
        radial stimulator positions to simulate.
    angles_rad: list of float;
        angle stimulator positions to simulate.
    temp_protocol: dict;
        temporal stim/recording protocol
        duration_ms: int;
        delay_ms: int;
        total_rec_time_ms: int;
    seg_rec_vars: list of 2 lists of str;
        [variable_names, variable_hoc_pointers], e.g.
        [['time [ms]', 'V_soma(0.5)'],
         ['h._ref_t', 'h.soma(0.5)._ref_v']] to record
         time in ms and voltage at soma in mV.
    allseg_rec_var: str/None;
        pointer which will be systematically applied to all segments, e.g.
        '._ref_g_cat_chanrhod' will record ChR conductance in all segs,
        variable names will be segment name.
    sim_data_transform: func(sim_data, segs);
       function must act on simulated data and segments (pandas DataFrame).
    scalar_result_names: list of str;
        Names of results to be generated.
    scalar_result_funcs: list of functions;
        Functions act on simulation results (pandas DataFrame) and segs.
    vector_result_func: function;
        Funciton actis on simdata, segs and returns Result as pandas DataFrame.
    interpol_dt_ms: float;
        interpolation time step.
    AP_threshold_mV: float;
        Threshold in mV to count spikes.

    Returns:
    --------
    pandas DataFrame;
        Spikes recorded at soma for various stim locations.
    """
    # NEURON setup
    h.load_file("stdrun.hoc")
    h.cvode_active(1)
    # load cell
    cell = Cell(
        hoc_file="simneurostim/model/hoc/" + cell_dict['cellname'] + ".hoc",
        cortical_depth=cell_dict['cortical_depth'],
        ChR_soma_density = cell_dict['ChR_soma_density'],
        ChR_distribution=cell_dict['ChR_distribution'],
        rm_mech_from_secs=None,
    )
    # create list of segments
    segs = [seg for sec in h.allsec() for seg in sec]
    if allseg_rec_var != None:
        # init recording variables
        varnames = [str(seg) for seg in segs]
        varpointers = ['h.'+str(seg)+allseg_rec_var for seg in segs]
        seg_rec_vars[0] += varnames
        seg_rec_vars[1] += varpointers
    # init stimulator
    stimulator = Stimulator(
        diameter_um=stimulator_dict['diameter_um'], 
        NA=stimulator_dict['NA'],
    ) 
    spatial_profile_df = simulate_spatial_profile_wo_NEURONsetup(
            cell, cell_dict, stimulator, segs, temp_protocol, stim_intensity_mWPERmm2, seg_rec_vars, interpol_dt_ms,
            sim_data_transform, scalar_result_names, scalar_result_funcs, vector_result_func, 
            radii_um, angles_rad
    )
    return spatial_profile_df

def simulate_spatial_profile_wo_NEURONsetup(
    cell, cell_dict, stimulator, segs, temp_protocol, stim_intensity_mWPERmm2, seg_rec_vars, interpol_dt_ms,
    sim_data_transform, scalar_result_names, scalar_result_funcs, vector_result_func, 
    radii_um, angles_rad, **kwargs
    ):
    """
    For docs see simulate_spatial_profile.
    """
    # simulate for all radius and angle combinations
    results = []
    for radius in radii_um:
        for angle in angles_rad:
            stim_x_um, stim_y_um = convert_polar_to_cartesian_xz(radius, angle)
            stim_z_um = 0  # cortical surface
            # init simulation
            simcontrol = SimControl(
                cell=cell,
                stimulator=stimulator
            )
            # run simulation
            try:
                sim_data = simcontrol.run(
                    temp_protocol=temp_protocol,
                    stim_location=(stim_x_um, stim_y_um, stim_z_um),
                    stim_intensity_mWPERmm2=stim_intensity_mWPERmm2,
                    rec_vars=seg_rec_vars,
                    interpol_dt_ms=interpol_dt_ms,
                )
                #print(sim_data.columns)
                if sim_data_transform != None:
                    # apply transformation to variables named str(seg)
                    sim_data_transform(sim_data, segs)
                scalar_results = [func(sim_data,segs) for func in scalar_result_funcs]
                if vector_result_func != None:
                    result = vector_result_func(sim_data, segs)
                    for name, scalar_result in zip(scalar_result_names, scalar_results):
                        result[name] = scalar_result
                else:
                    result = pd.DataFrame(columns=scalar_result_names, data=scalar_results)
                result['RuntimeError'] = False
            except RuntimeError:
                result['RuntimeError'] = True
            # define index of result dataframe
            result["hoc_file"] = cell_dict['cellname']
            result["chanrhod_distribution"] = cell_dict['ChR_distribution']
            result["chanrhod_expression"] = cell_dict['ChR_soma_density']
            result["light_model"] = stimulator.modelname
            result["fiber_diameter"] = stimulator.diameter_um
            result["fiber_NA"] = stimulator.NA
            result["light_power"] = \
                stim_intensity_mWPERmm2*(stimulator.diameter_um/2*1e-3)**2*np.pi*1e-3
            result["stim_duration [ms]"] = temp_protocol['duration_ms']
            result["radius [um]"] = radius
            result["angle [rad]"] = angle
            results.append(result)
    # concatenate and save data
    spatial_profile_df = pd.concat(results).set_index(
        [
            "hoc_file",
            "light_model",
            "chanrhod_distribution",
            "chanrhod_expression",
            "fiber_diameter",
            "fiber_NA",
            "stim_duration [ms]",
            "light_power",
            "radius [um]",
            "angle [rad]",
        ]
    )
    spatial_profile_df['RuntimeError'] = spatial_profile_df['RuntimeError'].astype(bool)
    return spatial_profile_df

def find_x_at_value(xarray, varray, value):
    if type(value) == np.ndarray:
        assert len(value)==1, "value is not unique."
        value = value[0]
    elif type(value) == list:
        assert len(value)==1, "value is not unique."
        value = value[0]
    varray = np.asarray(varray)
    # check if there is a proper stimulation otherwise return x_value = np.nan
    if np.any(varray>value):
        # take rightmost index where varray is larger than value
        idx = np.where(varray>value)[0][-1]
        # interpolate between this value and the next
        # (where varray has to be smaller/equal to value)
        x1 = xarray[idx]
        v1 = varray[idx]
        x2 = xarray[idx+1]
        v2 = varray[idx+1]
        x_value = x1 + (value-v1) * (x2-x1) / (v2-v1)
        return x_value
    else:
        return np.nan

def calc_pr_rsc(spatial_profile_df, groupby=None):
    """Calculate pr and rsc for given spatial profile data frame"""

    if groupby==None:
        groupby = ['hoc_file', 'light_model','chanrhod_distribution','chanrhod_expression',
                   'fiber_diameter', 'fiber_NA', 'stim_duration [ms]','light_power']
    # average over angular stimulator coordinates:
    radial_profile_df = spatial_profile_df.groupby(groupby+['radius [um]']).mean()
    # calc peak reponse (maximum of radial profile)
    pr_rsc_df = pd.DataFrame(radial_profile_df.groupby(groupby)["AP_count"].max()).rename(columns=dict(AP_count='peak_response'))
    # calc response space constant 
    pr_rsc_df['response_space_constant_um'] = radial_profile_df.reset_index().merge(
        pr_rsc_df, on=groupby, how='left'
        ).groupby(groupby)[["radius [um]","AP_count","peak_response"]].apply(
        lambda x: find_x_at_value(
            xarray=x['radius [um]'].values,
            varray=x['AP_count'].values,
            value=x['peak_response'].unique()/2
        )
    )
    return pr_rsc_df

def find_target_pr(
        target_pr, tolerance, stim_intensity_mWPERmm2_minmax, 
        simulate_spatial_profile_args):

    # setup NEURON, cell, and stimulator to enable simulation of spatial profiles
    # NEURON setup
    h.load_file("stdrun.hoc")
    h.cvode_active(1)
    # load cell
    cell_dict = simulate_spatial_profile_args['cell_dict']
    cell = Cell(
        hoc_file="simneurostim/model/hoc/" + cell_dict['cellname'] + ".hoc",
        cortical_depth=cell_dict['cortical_depth'],
        ChR_soma_density = cell_dict['ChR_soma_density'],
        ChR_distribution=cell_dict['ChR_distribution'],
        rm_mech_from_secs=None,
    )
    # init stimulator
    stimulator_dict=simulate_spatial_profile_args['stimulator_dict']
    stimulator = Stimulator(
        diameter_um=stimulator_dict['diameter_um'], 
        NA=stimulator_dict['NA'],
    ) 

    # actual search algorithm
    pr_not_found = True
    min_si, max_si = stim_intensity_mWPERmm2_minmax
    cnt = 0
    while pr_not_found:
        # define test stimulation intensity (mW/mm2)
        test_si = np.exp(0.5 * (np.log(min_si) + np.log(max_si)))
        simulate_spatial_profile_args['stim_intensity_mWPERmm2'] = test_si
        spatial_profile = simulate_spatial_profile_wo_NEURONsetup(
            cell=cell, stimulator=stimulator, segs=None, 
            **simulate_spatial_profile_args
        )
        pr_rsc_df = calc_pr_rsc(spatial_profile, groupby=None)
        pr = pr_rsc_df.peak_response.values[0]
        print("tested ", test_si, " and found: pr=",pr)
        if pr >= target_pr*(1-tolerance) and pr <= target_pr*(1+tolerance):
            break
        elif pr > target_pr*(1+tolerance):
            max_si = test_si
        elif pr < target_pr*(1-tolerance):
            min_si = test_si
        cnt += 1
        if cnt > 2:
            print("found no target peak response")
            test_si = np.nan
            break
    return test_si
