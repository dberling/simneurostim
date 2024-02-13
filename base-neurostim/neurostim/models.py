import subprocess
from neuron import h
from neurostim.utils import arbitrary_3d_rotation_along_axis
import numpy as np

def compile_mod_files(mod_dir):
    """
    Compile NEURON mod files.
    """
    subprocess.run(" ".join(["nrnivmodl",mod_dir]), shell=True)
    
class CellModelTemplate():
    """
    Template to initialize cell model.

    Params:
    -------
    cortical_depth: float
       depth of soma section measured from surface.
    rotation: dict(axis: str, angle: float)
       axis and angle to rotate cell
    insert_ChR: bool
       whether to insert ChR
    soma_sec: str
       name of soma section
    """
    def __init__(
        self,
        cortical_depth,
        rotation,
        insert_ChR,
        soma_sec
        ):
        self.soma_sec = eval('h.'+soma_sec)
        self.cortical_depth = cortical_depth
        self.rotation = rotation
        self._rotate()
        self._move_to_cortical_depth()
        if insert_ChR:
            self._insert_ChR()

    def _rotate(self):
        """
        Rotate cell
        """
        axis = self.rotation['axis']
        angle = self.rotation['angle']
        for sec in h.allsec():
            for i in range(sec.n3d()):
                pos = [sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                rot_pos = arbitrary_3d_rotation_along_axis(pos, axis, angle)
                h.pt3dchange(i, *rot_pos, sec.diam3d(i), sec=sec)

    def _move_to_cortical_depth(self):
        """
        Move cell to cortical depth measured at soma.
        """
        cortical_depth = self.cortical_depth
        for sec in h.allsec():
            for i in range(sec.n3d()):
                cortex_pos = [sec.x3d(i), sec.y3d(i), sec.z3d(i) - self.cortical_depth]
                h.pt3dchange(i, *cortex_pos, sec.diam3d(i), sec=sec)

    def _insert_ChR(self):
        """
        Insert Channelrhodopsin mechanisms into all sections.
        """
        for sec in h.allsec():
            sec.insert('chanrhod')

def L5_catV1():
    """
    Model code by Foutz et al 2012, Hu et al. 2009, ..., Mainen & Sejnowski 1996
    """
    # compile corresponding mod files
    # compile_mod_files('simneurostim/model/mod/foutz2012')
    # initialize model in NEURON
    h.load_file('simneurostim/model/hoc/L5.hoc')
    # use CellModelTemplate to define cell properties
    cell = CellModelTemplate(
            cortical_depth = 1170, #um
            rotation = dict(
                axis="y",
                angle=np.pi/2
            ),
            insert_ChR=False,
            soma_sec='soma'
    )
    cell.modelname = 'L5_catV1'
    return cell

def L23_catV1():
    """
    Model code by Foutz et al 2012, Hu et al. 2009, ..., Mainen & Sejnowski 1996
    """
    # compile corresponding mod files (doesn't work as env needs to be reloaded to work)
    # compile_mod_files('simneurostim/model/mod/foutz2012')
    # initialize model in NEURON
    h.load_file('simneurostim/model/hoc/L23.hoc')
    # use CellModelTemplate to define cell properties
    cell = CellModelTemplate(
            cortical_depth = 400, #um
            rotation = dict(
                axis="x",
                angle=np.pi/2
            ),
            insert_ChR=False,
            soma_sec='soma'
    )
    cell.modelname = 'L23_catV1'
    return cell

def L5_Hay2011():
    """
    Model code associated with Hay et al., PLoS Computational Biology, 2011
    """
    # compile corresponding mod files
    #compile_mod_files('simneurostim/model/mod/hay2011')
    # initialize model in NEURON
    h.load_file('import3d.hoc')
    h.load_file('simneurostim/model/hoc/HayEtAl2011/L5PCbiophys3.hoc')
    h.load_file('simneurostim/model/hoc/HayEtAl2011/L5PCtemplate.hoc')
    h_model_object = h.L5PCtemplate('simneurostim/model/hoc/HayEtAl2011/morphologies/cell1.asc')

    # use CellModelTemplate to define cell properties
    cell = CellModelTemplate(
            cortical_depth = 1200, #um
            rotation = dict(
                axis="x",
                angle=np.pi/2
            ),
            insert_ChR=True,
            soma_sec='L5PCtemplate[0].soma[0]'
    )
    cell.modelname = 'L5_Hay2011'
    return cell, h_model_object
