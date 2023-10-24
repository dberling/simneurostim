import neurostim.light_propagation_models as light_propagation_models

class Stimulator():
    def __init__(self, diameter_um, NA):
        """
        Optical fiber light model implemented according to 
        Foutz et al. 2012, Aravnis et al. 2007.
        """
        self.diameter_um = diameter_um
        self.NA = NA
        self.light_propagation = getattr(light_propagation_models, "fiber_Foutz2012")
        self.modelname="foutz_et_al2012"

    def calculate_Tx_at_pos(self, pos_xyz_um, stim_xyz_um):
        """
        Compute light loss at position pos_xyz_um considering stimulator at stim_xyz_um.

        Args:
            pos_xyz_um (list): (x, y, z) position of the target in um
            stim_xyz_um (list): (x, y, z) position of the stimulator in um

        Returns:
            float: loss factor [1/cm2]
        """
        pos_xyz_um = [target - source for target, source in zip(pos_xyz_um, stim_xyz_um)]
        Tx = self.light_propagation(
            *pos_xyz_um, self.diameter_um, NA=self.NA
        )
        return Tx
