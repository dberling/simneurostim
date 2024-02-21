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
    soma_xyz: [x:float, y:float, z:float]
        cortical position of the soma in um.
    rotation: dict(axis: str, angle: float)
        axis and angle to rotate cell
    insert_ChR: bool
        whether to insert ChR
    soma_sec: str
        name of soma section
    """
    def __init__(
        self,
        soma_xyz,
        rotation,
        insert_ChR,
        soma_sec
        ):
        self.soma_sec = eval('h.'+soma_sec)
        self.rotation = rotation
        self._rotate()
        self._move_to_soma_position(soma_xyz=soma_xyz)
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

    def _move_to_soma_position(self,soma_xyz):
        """
        Move cell to cortical depth measured at soma.

        Params:
        -------
        soma_xyz: [x:float, y:float, z:float]
            cortical position of the soma in um.
        """
        # move all secs to positions such that soma at position
        x, y, z = soma_xyz
        # original position:
        xo, yo, zo = [self.soma_sec.x3d(0), self.soma_sec.y3d(0), self.soma_sec.z3d(0)]
        # move by
        dx, dy, dz = [xo-x, yo-y, zo-z]
        for sec in h.allsec():
            for i in range(sec.n3d()):
                cortex_pos = [sec.x3d(i) - dx, sec.y3d(i) - dy, sec.z3d(i) - dz]
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
            soma_xyz = [0,0,-1170], #um
            rotation = dict(
                axis="y",
                angle=np.pi/2
            ),
            insert_ChR=False,
            soma_sec='soma'
    )
    cell.modelname = 'L5_catV1'
    return cell, None

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
            soma_xyz = [0,0,-400], #um
            rotation = dict(
                axis="x",
                angle=np.pi/2
            ),
            insert_ChR=False,
            soma_sec='soma'
    )
    cell.modelname = 'L23_catV1'
    return cell, None

def L5_Hay2011(cell):
    """
    Model code associated with Hay et al., PLoS Computational Biology, 2011

    Params:
    -------
    cell: str
        ('cell1', 'cell2', 'cell3')

    """
    # compile corresponding mod files
    #compile_mod_files('simneurostim/model/mod/hay2011')
    # initialize model in NEURON
    h.load_file('import3d.hoc')
    h.load_file('simneurostim/model/hoc/HayEtAl2011/L5PCbiophys3.hoc')
    h.load_file('simneurostim/model/hoc/HayEtAl2011/L5PCtemplate.hoc')
    h_model_object = h.L5PCtemplate('simneurostim/model/hoc/HayEtAl2011/morphologies/'+cell+'.asc')

    if cell == 'cell1':
        soma_z = -1200
    elif cell == 'cell2':
        soma_z = -1100
    elif cell == 'cell3':
        raise NotImplementedError("Need to define soma depth for cell3 morphology")

    # use CellModelTemplate to define cell properties
    cell = CellModelTemplate(
            soma_xyz = [0, 0, soma_z], #um
            rotation = dict(
                axis="x",
                angle=np.pi/2
            ),
            insert_ChR=True,
            soma_sec='L5PCtemplate[0].soma[0]'
    )
    cell.modelname = 'L5_Hay2011'
    return cell, h_model_object

def L5_Hay2011_cell1():
    return L5_Hay2011(cell='cell1')
def L5_Hay2011_cell2():
    return L5_Hay2011(cell='cell2')
def L5_Hay2011_cell2_vertical_shaft():
    cell, _ = L5_Hay2011(cell='cell2')
    cell.rotation = dict(
            axis = 'y',
            angle = -1*np.pi/16
    )
    cell._rotate()
    cell._move_to_soma_position(soma_xyz=[0, 0, -1100])
    return cell, _
def L5_Hay2011_cell2_vertical_shaft_10higherapicsod():
    cell, hoc_obj = L5_Hay2011_cell2_vertical_shaft()
    for sec in h.allsec():
        if 'apic' in str(sec) and sec.has_membrane('NaTa_t'):
            sec.gNaTa_tbar_NaTa_t *= 10
    return cell, hoc_obj
