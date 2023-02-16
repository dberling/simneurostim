from neuron import h
import numpy as np
import neurostim.light_propagation_models as light_propagation_models
from scipy.interpolate import interp1d
import pandas as pd


class LightSource:
    def __init__(self, model, position, width, NA=None):
        """Light model class. Contains model of light propagation,
            position and width of the light source

        Args:
            model (str): light model name (to match function name in optostim.light_models)
            position (list of floats): (x, y, z) position of the light source
            width (float): width of the light source
            NA (float): numerical aperture, only valid param for yona_et_al and foutz light source
        """
        self.name = model
        self.light_propagation = getattr(light_propagation_models, model)
        self.width = width
        self.NA = NA
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        if self.name == "soltan_et_al2017":
            df = pd.read_csv("metadata/light_angle_profile_Soltan_et_al.csv")
            f = interp1d(
                df["alpha_radiants"], df["normalized_intensity"], kind="linear"
            )
            n_alphas = int(1e6)
            alphas = np.linspace(0, np.pi / 2, n_alphas)
            self.intensity_angle_profile = f(alphas)
        elif self.name == "yona_et_al2016":
            self.intensity_angle_profile = dict(
                    I=np.load("".join([
                        "metadata/yona_I_NA",
                        str(NA),
                        ".npy"])),
                    rhorho=np.load("".join([
                        "metadata/yona_rhorho_NA",
                        str(NA),
                        ".npy"])),
                    zz=np.load("".join([
                        "metadata/yona_zz_NA",
                        str(NA),
                        ".npy"]))
            )
        else:
            self.intensity_angle_profile = None

    def calculate_Tx_at_pos(self, xyz):
        """Compute transfer (dampening) factor of the light
            irradiance at target position xyz.

        Args:
            xyz (list): (x, y, z) position of the target

        Returns:
            float: transfer factor [1/cm2]
        """
        xyz = [target - source for target, source in zip(xyz, self.position)]
        Tx = self.light_propagation(
            *xyz, self.width, power=1, intensity_profile=self.intensity_angle_profile, NA=self.NA
        )
        return Tx

class LightStimulation:
    def __init__(self, cell, light_source, delay, duration, light_power, record_all_segments):
        """Light stimulation class

        Args:
            cell (Cell): cell to stimulate
            light_source (LightSource): light source model
            delay (float): stimulation onset time in ms
            duration (float): duration of stimulation in ms
            light_power (float): power of light source [W]
        Details on the light power calculations in stimulation:
        The light_power [W] is passed to the parameter "amp" in ostim.mod (mod-file) which
        keeps track of the light source power [W]. In ostim.mod, the number of photons per time
        emitted by the light source is calculated ("flux", unit [1/ms]). This parameter is 
        accessed in chanrhod.mod as "source_flux" and converted into the number photons 
        propagating to the location where the segment is located by multiplication with "Tx", 
        in some places called transfer resistance and representing the fraction of photons 
        reaching the segment per unit area (1/cm2) from the photons exiting the fiber. This 
        parameter is called in chanrhod.mod "flux" with unit [1/ms/cm2]. This "flux" [1/ms/cm2]
        is multiplied with the crossection of channelrhodopsin ("sigma_retinal", [cm2] to obtain
        the number of photons hitting the channel per time, parameter called "phi" in chanrhod.mod.
        unit is [1/ms]. 
        The Tx parameter described the loss of photons due to propagation in the cortical medium 
        and is calculated using one of the light models and setting the light_power to 1W. This
        way the loss is relative and the light_power can be multiplied to achieve the actual light
        intensity at a specific location in space (light_power * Tx). This is possible because the
        light intensity is linear to the employed light power. This calculation takes place in
        chanrhod.mod when the "flux" is calculated form the "source_flux": 
        flux = source_flux * Tx
        """
        self.cell = cell
        self.light_source = light_source
        self.sec = h.Section(name="Light_source: " + light_source.name)
        self.stim = h.ostim(0.5, self.sec)
        self.stim.amp = light_power
        self.stim.dur = duration
        self.stim.delay = delay
        # Connects variables in ostim mod file to variables in chanrhod mod file
        h.setpointer(h._ref_source_irradiance_chanrhod, "irradiance", self.stim)
        h.setpointer(h._ref_source_flux_chanrhod, "flux", self.stim)
        h.setpointer(h._ref_tstimon_chanrhod, "tstimon", self.stim)
        h.setpointer(h._ref_tstimoff_chanrhod, "tstimoff", self.stim)
        self.rec_var_pointers_dict = self.cell.get_rec_variables_pointers_dict(
            record_all_segments=record_all_segments
        )

    def get_segs_Txs(self):
        """returns a 2-tuple containing NEURON segments and corresponding Txs"""
        segs = [
            seg
            for sec in h.allsec()
            if h.ismembrane("chanrhod", sec=sec)
            for seg in sec
        ]
        Txs = [
            self.light_source.calculate_Tx_at_pos(
                (seg.x_chanrhod, seg.y_chanrhod, seg.z_chanrhod)
            )
            for sec in h.allsec()
            if h.ismembrane("chanrhod", sec=sec)
            for seg in sec
        ]
        return (segs, Txs)

    def assign_pos_chanrhod(self):
        """Assign x, y, and z chanrhod to neuron section"""
        # TODO figure out why I need to interpolate
        for sec in list(h.allsec()):
            if h.ismembrane("chanrhod", sec=sec):
                n = sec.n3d()
                x = h.Vector(n)
                y = h.Vector(n)
                z = h.Vector(n)
                len = h.Vector(n)
                for i in range(n):
                    x.x[i] = sec.x3d(i)
                    y.x[i] = sec.y3d(i)
                    z.x[i] = sec.z3d(i)
                    len.x[i] = sec.arc3d(i)
                len.div(len.x[n - 1])
                r = h.Vector(sec.nseg + 2)
                r.indgen(1.0 / sec.nseg)
                r.sub(1.0 / (2.0 * sec.nseg))
                r.x[0] = 0
                r.x[sec.nseg + 1] = 1
                x_int = h.Vector(sec.nseg + 2)
                y_int = h.Vector(sec.nseg + 2)
                z_int = h.Vector(sec.nseg + 2)
                x_int.interpolate(r, len, x)
                y_int.interpolate(r, len, y)
                z_int.interpolate(r, len, z)
                for i in range(1, sec.nseg + 1):
                    xr = r.x[i]
                    sec(xr).x_chanrhod = x_int.x[i]
                    sec(xr).y_chanrhod = y_int.x[i]
                    sec(xr).z_chanrhod = z_int.x[i]

    def simulate_and_measure(self, tot_rec_time=200,
            extra_rec_var_names=[], extra_rec_var_pointers=[]):
        """Simulate and measure relevant NEURON variables.
            Which exactly are these variables depends on the cell,

        Args:
            tot_rec_time (int, optional): Defaults to 200.

        Returns:
            dict: Dictorary of variables measurements.
                  Each key corresponds to a recorded variable
        """
        ChR2_segs, Txs = self.get_segs_Txs()
        for seg, Tx in zip(ChR2_segs, Txs):
            seg.Tx_chanrhod = Tx
        # initialize dictionary of measurements of relevant variables
        # which are specific to the cell (this method use the dictionary
        # of pointers to variables of the cell object)
        rec_var_measures_dict = {}
        for var_name, var_pointer in self.rec_var_pointers_dict.items():
            rec_var_measures_dict[var_name] = h.Vector()
            rec_var_measures_dict[var_name].record(var_pointer)
        if extra_rec_var_names:
            for var_name, var_pointer in zip(extra_rec_var_names, extra_rec_var_pointers):
                rec_var_measures_dict[var_name] = h.Vector()
                rec_var_measures_dict[var_name].record(var_pointer)
        assert (
            tot_rec_time > self.stim.delay + self.stim.dur
        ), "recording stops before stimulation ends"
        h.tstop = tot_rec_time
        # TODO could make a parameter of the simulation
        h.v_init = -71.0442766389535
        h.run()
        for var_name, var_measures in rec_var_measures_dict.items():
            rec_var_measures_dict[var_name] = np.array(var_measures)
        return rec_var_measures_dict

    def get_currents_components_into_soma(self, rec_var_measures_dict):
        """Compute currents components into soma given dictionary of measurements

        Args:
            rec_var_measures_dict (dict):
                dictionary of measurements obtained with self.simulate_and_measure()

        Returns:
            dict: dictionary of currents components into the soma
        """
        i_components_into_soma = {
            "time [ms]": rec_var_measures_dict["time [ms]"],
            "V_soma(0.5)": rec_var_measures_dict["V_soma(0.5)"],
            "i_ChR2_into_soma(0.5)": -rec_var_measures_dict["i_ChR2_soma(0.5)"],
        }
        for soma_childseg, parentseg in self.cell.soma_child_relations:
            name = "i_" + str(soma_childseg) + "_into_soma"
            dV = (
                rec_var_measures_dict["V_" + str(parentseg)]
                - rec_var_measures_dict["V_" + str(soma_childseg)]
            )
            i = -dV / eval("h." + str(soma_childseg) + ".ri()")
            i_components_into_soma[name] = i
        i_list = [v for k, v in i_components_into_soma.items() if k.startswith("i_")]
        i_components_into_soma["i_tot_into_soma"] = np.sum(i_list, axis=0)
        return i_components_into_soma

    # def get_secs_names_for_axial_currents(self):
    #     """ Get name of segments connected to soma. Useful for compute axial currents"""
    #     names = self.rec_var_pointers_dict.keys()
    #     names = [name for name in names if name.startswith("Vax_")]
    #     return names

    def simulate_and_measure_current_components_into_soma(self, tot_rec_time):
        """Run simulation and return temporal profiles of different current
            components into soma.

        Args:
            tot_rec_time (int, optional): Defaults to 200.

        Returns:
            dict: temporal profiles of the current components into soma
        """
        meas_dict = self.simulate_and_measure(tot_rec_time)
        return self.get_currents_components_into_soma(meas_dict)
