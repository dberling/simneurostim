from neuron import h
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

class SimControl:
    def __init__(self, cell, stimulator):
        """
        Initialize simulation with cell and stimulator.
        """
        self.cell = cell
        # assert that cell is placed below cortical surface
        assert(
            np.max([sec.z3d(i) for sec in cell.sections for i in range(sec.n3d())]) < 0,
            "Parts of the cell are placed above the cortical surface!"
        )
        self.stimulator = stimulator
        # initialize stimualtor in NEURON as section and connect to mod file ostim.mod
        self.stim = h.ostim(0.5, h.Section(name="stimulator"))
        h.setpointer(h._ref_source_flux_chanrhod, "flux", self.stim) # not sure if needed
        h.setpointer(h._ref_tstimon_chanrhod, "tstimon", self.stim) # not sure if needed
        h.setpointer(h._ref_tstimoff_chanrhod, "tstimoff", self.stim) # not sure if needed

    def get_segs_Txs(self, stim_location):
        """
        Calculate the light losses for each NEURON segment.
        
        Parameters:
        -----------
        stim_location: 3-tuple of float;
            Stimultor location: (stim_x_um, stim_y_um, stim_z_um).

        Returns: 2-tuple;
            (NEURON segments, corresponding Txs)
        """

        segs = [
            seg 
            for sec in h.allsec()
            if h.ismembrane("chanrhod", sec=sec)
            for seg in sec
        ]
        Txs = [
            self.stimulator.calculate_Tx_at_pos(
                pos_xyz_um=(seg.x_chanrhod, seg.y_chanrhod, seg.z_chanrhod),
                stim_xyz_um=stim_location,
            )
            for sec in h.allsec()
            if h.ismembrane("chanrhod", sec=sec)
            for seg in sec
        ]
        return (segs, Txs)

    def _interpolate(self, df, interpolation_dt):
        time = np.arange(df["time [ms]"].min(), df["time [ms]"].max(), interpolation_dt)
        df_int = pd.DataFrame({})
        for name, values in df.items():
            f = interp1d(df["time [ms]"], values)
            df_int[name] = f(time)
        return df_int

    def _postprocess(self, simdict, interpol_dt_ms):
        sim_data = pd.DataFrame(simdict)
        # dealing with drop full row duplicates
        # drop completely redundant duplicates
        sim_data = sim_data.drop_duplicates()
        # add 1e-12 ms to 2nd entry time point of duplicate
        # entries with the same time but different (e.g. Vm) values
        sim_data.loc[sim_data["time [ms]"].diff() == 0, "time [ms]"] += 1e-12
        # interpolate simulation results
        sim_data = self._interpolate(
            df=sim_data, interpolation_dt=float(interpol_dt_ms)
        )
        return sim_data

    def run(self, 
            temp_protocol, 
            stim_location, 
            stim_intensity_mWPERmm2, 
            rec_vars, 
            interpol_dt_ms,
            norm_power_mW_of_MultiStimulator=None
            ):
        """
        Run simulation.

        Args:
        -----
        temp_protocol: dict;
            Dictionary containing "duration_ms", "delay_ms", "total_rec_time_ms"
        stim_location: tuple;
            Tuple containing (x_um, y_um, z_um) of stimulator in um
        stim_intensity_mWPERmm2: float/None;
            Stimulation intensity at stimulator output surface in mW/mm2.
            Works only for single stimulator. Should be None for MultiStimulator.
        rec_vars: tuple;
            Tuple containing recording variable names and pointers
            (variable_names (List of str), variable_pointers (List of hoc pointers)
        interpol_dt_ms: float;
            time step to interpolate results with.
        norm_power_mW_of_MultiStimulator: float/None;
            Power which is applied to each stimulator of a MultiStimulator and
            weighted by the intensity_scale variable given for each stimulator. 

        Returns Pandas DataFrame with recordings
        """
        assert len(rec_vars[0]) == len(rec_vars[1]), "Recording variable names and pointers don't have the same dimenstion"
        # communicate stim_intensity to ostim.mod
        if stim_intensity_mWPERmm2 != None:
            assert norm_power_mW_of_MultiStimulator == None,\
                "If using single stimulator, norm_power_mW_of_MultiStimulator should be set to None\
                 Use stim_intensity_mWPERmm2 instead to set light intensity."
            # assume single stimulator
            # convert to power of the stimulator [W]:
            self.stim.power_W = stim_intensity_mWPERmm2\
                * (self.stimulator.diameter_um/2*1e-3)**2*np.pi*1e-3
        elif norm_power_mW_of_MultiStimulator != None:
            assert stim_intensity_mWPERmm2 == None,\
                "If using MultiStimulator, stim_intensity_mWPERmm2 should be set to None.\
                 Use norm_power_mW_of_MultiStimulator instead to light power."
            # assume MultiStimulator:
            self.stim.power_W = norm_power_mW_of_MultiStimulator * 1e-3

        self.stim.dur = temp_protocol['duration_ms']
        self.stim.delay = temp_protocol['delay_ms']
        # calculate light intensity at segments
        ChR_segs, Txs = self.get_segs_Txs(stim_location)
        for seg, Tx in zip(ChR_segs, Txs):
            seg.Tx_chanrhod = Tx
        # initialize dictionary of measurements
        rec_var_measures_dict = {}
        for var_name, var_pointer in zip(rec_vars[0], rec_vars[1]):
            rec_var_measures_dict[var_name] = h.Vector()
            rec_var_measures_dict[var_name].record(var_pointer)
        h.tstop = temp_protocol['total_rec_time_ms']
        h.v_init = -71.0442766389535 #mV
        h.run()
        for var_name, var_measures in rec_var_measures_dict.items():
            rec_var_measures_dict[var_name] = np.array(var_measures)

        return self._postprocess(rec_var_measures_dict, interpol_dt_ms)

