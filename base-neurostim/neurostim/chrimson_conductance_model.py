import numpy as np
from neurostim.stimulator import MultiStimulator
import ast
from scipy.constants import h, c
import math
from copy import copy

def ChRsystem_step(state: np.ndarray, I: float, sampling_period: float) -> np.ndarray:
    """ Photon flux I must be given in photons/cm2/fs (femto-second 1e-15)"""
    PhoC1toO1_times_I = 1.0993e-4 * 50 * I
    PhoC2toO2_times_I = 7.1973e-5 * 50 * I
    PhoC1toC2_times_I = 1.936e-6  * 50 * I
    PhoC2toC1_times_I = 1.438e-5  * 50 * I

    O1toC1 = 0.125
    O2toC2 = 0.015
    O2toS  = 0.0001
    C2toC1 = 1e-7
    StoC1  = 3e-6

    O1,O2,C1,C2,S = state

    _O1 = - O1toC1 * O1                    + PhoC1toO1_times_I * C1
    _O2 = - O2toC2 * O2                    + PhoC2toO2_times_I * C2            - O2toS * O2

    _S  = (- StoC1 * S + O2toS * O2) * np.ones_like(I)

    _C1 = O1toC1 * O1    - PhoC1toO1_times_I * C1  - PhoC1toC2_times_I * C1   + C2toC1 * C2             + PhoC2toC1_times_I * C2  + StoC1 * S
    _C2 = O2toC2 * O2    - C2toC1 * C2             - PhoC2toC1_times_I * C2   + PhoC1toC2_times_I * C1  - PhoC2toO2_times_I * C2

    # print([format(number, ".2e") for number in[_O1,_O2,_C1,_C2,_S]])
    #print(type(_O1))

    return np.array([_O1,_O2,_C1,_C2,_S]) * sampling_period

def ChR_integration(y0, intensities, sampling_period):
    """
    Intensities contains intensity sampled at sampling_period.

    Initial state 0 must have shape (5, N), describing the fraction
    of molecules in states O1, O2, C1, C2, S; and for N instances 
    of a molecules that receive different intensity.
    Intensities must have shape (T, N) for T times and N instances
    of molecules.

    Intensity must be given in photons/cm2/fs [femto!], specific time
    unit is to avoid floating point arrays in numpy due to large numbers.
    """
    states = []
    state = y0.copy()
    for intensity in intensities:
        state += ChRsystem_step(state, I=intensity, sampling_period=sampling_period)
        states.append(copy(state))
    return np.array(states)

def calc_ChR_conductances_forloop(intensities, sampling_period):
    """
    Helper function to verify that numpy calculations (below) are correct.
    """
    # ChR initial state: 80% in closed 1 and 20% in closed 2
    y0 = np.array([0., 0., 0.8, 0.2, 0.0]) 
    conds = []
    for intensity_over_time in intensities:
        channel_states = ChR_integration(
            y0=y0, 
            intensities=intensity_over_time,
            sampling_period=sampling_period
        )
        O1 = channel_states[:,0]
        O2 = channel_states[:,1]
        # conductance in state O1 50fS (Foutz for ChR2) and O2 8.5fS (ratio O2/O1=0.17 according to Antolik 2021)
        # fs = 1e-15, muliplied with 1e9 to obtain nS -> 1e-6
        channel_conductance_nS = O1 * 50e-6 + O2 * 8.5e-6
        conds.append(channel_conductance_nS)
    return conds
    
def calc_ChR_conductances_numpy(intensities, sampling_period):
    """
    Returns single-channel Chrimson conductance in nS.
    """
    y0 = np.array([0., 0., 0.8, 0.2, 0.0]) 
    y0 = np.tile(y0, (intensities.shape[1],1)).T
    
    channel_states = ChR_integration(
            y0=y0, 
            intensities=intensities,
            sampling_period=sampling_period
        )
    O1 = channel_states[:,0,:]
    O2 = channel_states[:,1,:]
    # conductance in state O1 50fS (Foutz for ChR2) and O2 8.5fS (ratio O2/O1=0.17 according to Antolik 2021)
    channel_conductance_nS = O1 * 50e-6 + O2 * 8.5e-6
    return channel_conductance_nS
    
def order_of_magnitude(number):
    if number == 0:
        return float('-inf')  # Logarithm of zero is undefined, return negative infinity
    return round(math.log10(abs(number)))
    
def define_sampling(intensities):
    """
    Input flux in photons/cm2/fs, receive sampling_period in ms.
    """
    sampling_period = 0.1
    int_max = np.max(intensities)
    if int_max > 1e3:
        sampling_period /= 10**(order_of_magnitude(int_max)) / 1e3
    return sampling_period
    
def calc_rescaled_comp_conductances_nS(
    norm_power_mW_of_MultiStimulator, 
    stimulator_config,
    comp_data, 
    temp_protocol,
    imp_diff,
    reject_if_sampling_smaller=0.001
):
    """
    Calculate ChrimsonR condcutance per neuron compartment rescaled to its effect at soma.

    Params:
    -------
    norm_power_mW_of_MultiStimulator: float
        norm_power of MultiStimulator in mW
    stimulator_config: list of dict
        MultiStimulator configuration
    comp_data: list
        data per compartment: secname, sec_x, transfer_resistance_MOhm, x, y, z, area_um2, channel_density_PERcm2
    temp_protocol: dict
        duration_ms: int, delay_ms: int, total_rec_time_ms: int
    imp_diff = ['dd_ds', 'dd_ss', 'dd_mean(ds,ss)']
        Whether impedance differense is calculated with transfer res. between
        soma and dendrite, input res. at soma, or mean of both.
    reject_if_sampling_smaller: float
        If sampling period falls below this number, reject calculation.

    Returns:
    --------
    rescaled_cond_nS: numpy.ndarray
        Rescaled conductances in shape (N_times, N_compartments)
    interpol_dt_ms: float
        sampling period in ms
    completed_flag: bool
    """
    # convert str(list) to real list if needed:
    if type(stimulator_config[0]['position']) == str:
        for config in stimulator_config:
            config['position'] = ast.literal_eval(config['position'])
    
    # calculate photon flux at light source output
    E_photon = h * c / (595e-9)
    photon_flux_source_PER_s = norm_power_mW_of_MultiStimulator * 1e-3 / E_photon # Power [J/s] / Photon_energy [J]
    photon_flux_source_PER_fs = photon_flux_source_PER_s * 1e-15
    
    # load neuron comp data
    secname, sec_x, input_resistance_MOhm, transfer_resistance_MOhm, x, y, z, area_um2, channel_density_PERcm2 = comp_data.T
    # since conductance is given in nS, convert resistances to GOhm
    input_resistance_GOhm = input_resistance_MOhm * 1e-3
    transfer_resistance_GOhm = transfer_resistance_MOhm * 1e-3
    soma_input_resistance_GOhm = input_resistance_GOhm[(secname==1) & (sec_x==0.5)].item()
    N_channel = area_um2 * 1e-8 * channel_density_PERcm2
    
    # load stimulator model
    stimulator = MultiStimulator(stimulator_config)
    
    # calculate light fluxes at compartments:
    fluxes_photons_PER_cm2_fs = [
        # flux at light source output [1/fs]
        photon_flux_source_PER_fs *\
        # returns combined light transmission in 1/cm2     
        stimulator.calculate_Tx_at_pos(
            pos_xyz_um =  [x_,y_,z_],
            stim_xyz_um = [0,0,0]
        ) for x_,y_,z_ in zip(x,y,z)
    ]
    
    # generate temporal evolution of stimulation
    interpol_dt_ms = define_sampling(fluxes_photons_PER_cm2_fs)
    if interpol_dt_ms < reject_if_sampling_smaller:
        return None, interpol_dt_ms, False
        # sampling period would be to small
    stimulation_times = np.ones(int(temp_protocol['total_rec_time_ms']/interpol_dt_ms))
    stimulation_times[:int(temp_protocol['delay_ms']/interpol_dt_ms)] = 0
    stimulation_times[int(temp_protocol['duration_ms']/interpol_dt_ms):] = 0
    
    fluxes_photons_PER_cm2_fs = [stimulation_times * flux for flux in fluxes_photons_PER_cm2_fs] 
    
    channel_conductance_nS = calc_ChR_conductances_numpy(
        intensities=np.array(fluxes_photons_PER_cm2_fs).T, 
        sampling_period=interpol_dt_ms
    )
    comp_conductance_nS = channel_conductance_nS * np.array(N_channel).reshape((1,len(secname)))
    transfer_resistance_GOhm = np.array(transfer_resistance_GOhm).reshape((1,len(secname)))
    input_resistance_GOhm = np.array(input_resistance_GOhm).reshape((1,len(secname)))
    if imp_diff = 'dd_ds':
        resistance_diff = np.abs(input_resistance_GOhm - transfer_resistance_GOhm)
    elif imp_diff = 'dd_ss':
        resistance_diff = np.abs(input_resistance_GOhm - soma_input_resistance_GOhm)
    elif imp_diff = 'dd_mean(ds,ss)':
        mean = (transfer_resistance_GOhm + soma_input_resistance) / 2
        resistance_diff = np.abs(input_resistance_GOhm - mean)

    rescaled_cond_nS = comp_conductance_nS / (1 + resistance_diff * comp_conductance_nS)
    return rescaled_cond_nS, interpol_dt_ms, True
