from neuron import h
import os
from neurostim.utils import arbitrary_3d_rotation_along_axis
import numpy as np


def rm_mech(mech, sec):
    mt = h.MechanismType(0)
    mt.select(mech)
    mt.remove(sec=sec)


class Cell:
    def __init__(
        self,
        hoc_file,
        cortical_depth=None,
        #n_ChR_channels=10354945,
        ChR_soma_density=13e9,
        ChR_distribution="uniform",
        rm_mech_from_secs=None,
        delete_all_secs_except_soma=False
    ):
        """
        Parameters:
        -----------

        hoc_file: str
            name of hoc file from which cell is loaded
        cortical_depth: dict
            Dictionary with name of cell ('L23' or 'L5') and corresponding depth of soma in cortical column measured from the surface
        ChR_soma_density: float
            ChR density in soma in 1/cm2, Foutz et al use 13e9 /cm2
        n_ChR_channels: int
            Number of ChR2 channels distributed over the neuron, default value is the number of channels distributed in the L5 cell from Foutz et al 2012 in the condition of a uniform ChR density of 13000000000 channels 1/cm2
        ChR_distribution: {'uniform', 'shemesh_fig1m_untrgtd', shemesh_fig1n_trgtd'}
            Type of distribution to distirbute the ChR channels over the morphology
        rm_mech_from_secs: [list of str, list of str]
            List containing on the first position a list of the mechanisms to remove, e.g. 'na' and secs from which these mechansisms are removed on the second position, e.g., 'h.soma'

        """
        self.hoc_file = hoc_file
        allowed_hoc = [
            "L23.hoc",
            "L23_noNa_in_soma.hoc",
            "L23OnlyChR2.hoc",
            "L23soma.hoc",
            "L23somaOnlyChR2.hoc",
            "L23somaWithoutNa.hoc",
            "L23WithoutNa.hoc",
            "L23WithoutNaAtSoma.hoc",
            "L4.hoc",
            "L5.hoc",
            "L5_noNa_in_soma.hoc",
        ]
        self.allowed_hoc_files = [os.path.join("simneurostim/model", "hoc", h) for h in allowed_hoc]
        self._construct_cell(hoc_file)
        self._rotate_in_vertical_position()
        self._move_to_cortical_position(cortical_depth)
        #self._distribute_ChR_channels(
        #    n_channels=n_ChR_channels, distribution=ChR_distribution
        #)
        self._distribute_ChR_density(
            ChR_soma_density=ChR_soma_density, distribution=ChR_distribution
        )
        self._assign_pos_chanrhod()
        self.segs_coord = self.get_segs_coord_dict()
        # self.dendrites = [
        #     s for s in h.allsec() if s.name() not in ["soma", "Soma", "SOMA"]
        # ]
        if delete_all_secs_except_soma:
            # delete all sections except of soma
            for sec in list(h.allsec()):
                if sec !=h.soma:
                    h.delete_section(sec=sec)
        self.soma_child_relations = self._get_soma_child_relations()

        if rm_mech_from_secs != None:
            for mech in rm_mech_from_secs[0]:
                for sec in rm_mech_from_secs[1]:
                    rm_mech(mech, eval(sec))

    def _construct_cell(self, hoc_file):
        h.load_file(hoc_file)

    def _assign_pos_chanrhod(self):
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

    def _get_soma_child_relations(self):
        """Get array of tuples with soma children segments and their parent segment.

        Returns:
            array of tuples: children segs on 1st pos, corresponding
                             soma segs on 2nd positon
        """
        assert (
            h.soma in h.allsec()
        ), "Construct cell before requesting children and soma relationships."
        soma_childsegs = [sec(0.001) for sec in h.soma.children()]
        parentsegs = [sec.parentseg() for sec in h.soma.children()]
        return list(zip(soma_childsegs, parentsegs))

    def _get_ChR_expression_level(self, distance_from_soma, distribution):
        """
        Defines expression distribution.
        Derivation of expression distributions from data of 
        Shemesh et al.(2017). Nature Neuroscience, 20(12), 1796–1806. 
        https://doi.org/10.1038/s41593-017-0018-8

        Derivation details in metadata/CoChR_expression_levels_Shemesh_et_al2021*.ipynb
        *: _cultures
           _slices

        """
        if distribution == "uniform":
            return 1
        elif distribution == "shemesh_fig1m_untrgtd":
            b = 0.019
            return np.exp(-b * (distance_from_soma))
        elif distribution == "shemesh_fig1n_trgtd":
            b = 0.110
            return np.exp(-b * (distance_from_soma))
        elif distribution == "shemesh_supfig9b_exp_yoff":
            b = 0.09819018
            y_off = 0.18541665
            res_expr_level = (1-y_off) * np.exp(-b*distance_from_soma) + y_off
            return (res_expr_level>0) * res_expr_level
        elif distribution == "shemesh_supfig9b_exp_lin_yoff":
            b = 0.12200734
            m=0.00086746
            y_off=0.25372033
            res_expr_level = (1-y_off) * np.exp(-b*distance_from_soma) - m * distance_from_soma + y_off
            return (res_expr_level>0) * res_expr_level
        else:
            raise ValueError(
                "distribution must equal {'uniform', 'shemesh_fig1m_untrgtd', shemesh_fig1n_trgtd'}"
            )
            return None

    def _distribute_ChR_density(self, ChR_soma_density, distribution):
        """
        Distribute channelrhodopsin through the morphology matching density at soma

        Params:
        -------

        ChR_density: float
            surface density of ChR in 1/cm2, Foutz et al. 2012 used 13e9 1/cm2
        distribution: {'uniform','shemesh_supfig9b_exp_yoff','shemesh_supfig9b_exp_lin_yoff'}
            derivation in 'metadata/CoChR_expression_levels_Shemesh_et_al2021.ipynb'.
                     
        """
        for sec in h.allsec():
            if h.ismembrane("chanrhod", sec=sec):
                for seg in sec:
                    distance_from_soma_center = h.distance(h.soma(0.5), seg)  # um
                    seg.channel_density_chanrhod = ChR_soma_density * self._get_ChR_expression_level(
                        distance_from_soma_center, distribution=distribution
                    )
        return None


    def _distribute_ChR_channels(self, n_channels, distribution):
        """
        Distribute channelrhodopsin throughout the morphology matching channels in the cell

        Parameters:
        -----------

        n_ChR_channels: int
            number of ChR2 channels to distribute

        distribution: {'uniform','shemesh_supfig9b_exp_yoff','shemesh_supfig9b_exp_lin_yoff'}
            derivation in 'metadata/CoChR_expression_levels_Shemesh_et_al2021*.ipynb'.
        """
        def _get_ChR_channels_in_list(seclist):
            channels = 0
            for sec in seclist:
                if h.ismembrane("chanrhod", sec=sec):
                    for seg in sec:
                        rho = seg.channel_density_chanrhod / 1e8  # 1/cm2 --> 1/um2
                        area = h.area(seg.x, sec=sec)  # um2
                        n = rho * area
                        channels += n
            return channels


        # get ChR expression level for each segment and count distributed channels
        distributed_channels = 0
        for sec in h.allsec():
            if h.ismembrane("chanrhod", sec=sec):
                for seg in sec:
                    distance_from_soma_center = h.distance(h.soma(0.5), seg)  # um
                    seg.channel_density_chanrhod = self._get_ChR_expression_level(
                        distance_from_soma_center, distribution=distribution
                    )
                    rho = seg.channel_density_chanrhod / 1e8  # 1/cm2 --> 1/um2
                    area = h.area(seg.x, sec=sec)  # um2
                    n = rho * area
                    distributed_channels += n
        norm_to_n_channels = n_channels / distributed_channels
        distributed_channels = 0
        for sec in h.allsec():
            if h.ismembrane("chanrhod", sec=sec):
                for seg in sec:
                    seg.channel_density_chanrhod *= norm_to_n_channels
                    rho = seg.channel_density_chanrhod / 1e8  # 1/cm2 --> 1/um2
                    area = h.area(seg.x, sec=sec)  # um2
                    n = rho * area
                    distributed_channels += n
        assert 0.001 > abs(
            distributed_channels - n_channels
        ), "distributed channels != n_channels"
        return None

    def get_rec_variables_pointers_dict(self, record_all_segments=False):
        """Allows to get a dictionary of pointers to the variables to record.
            Such variables depends on the cell hoc file

        Raises:
            ValueError: if cell hoc file is not among the one in the list

        Returns:
            dict: dictionary of pointers to NEURON variables to record
        """
        var_names = ["time [ms]", "V_soma(0.5)", "i_ChR2_soma(0.5)"]
        var_pointers = [
            h._ref_t,
            h.soma(0.5)._ref_v,
            h.soma(0.5)._ref_i_chanrhod_chanrhod,
        ]
        if record_all_segments:
            for sec in h.allsec():
                for seg in sec:
                    var_names.append("V_" + str(seg))
                    var_pointers.append(seg._ref_v)
                    if h.ismembrane("chanrhod", sec=sec):
                        var_names.append("i_ChR2_" + str(seg))
                        var_pointers.append(seg._ref_i_chanrhod_chanrhod)
        if len(self.soma_child_relations) > 0:
            # add voltage recordings of soma child segs and their parentseg
            for soma_childseg, parentseg in self.soma_child_relations:
                var_names.append("V_" + str(soma_childseg))
                var_pointers.append(soma_childseg._ref_v)
                if "V_" + str(parentseg) not in var_names:
                    var_names.append("V_" + str(parentseg))
                    var_pointers.append(parentseg._ref_v)

        var_pointers_dict = dict(zip(var_names, var_pointers))
        return var_pointers_dict

    def get_segs_coord_dict(self):
        """get dictionary of segments extremes/points inside the sections

        Returns:
            dict : key is the section name, value is a list of 3 entries list
            containing the coordinates (x,y,z) of each point in the section
        """
        segs_coord = {}
        for sec in list(h.allsec()):
            segs_coord[sec.name()] = [
                [sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())
            ]
        return segs_coord

    def _rotate_in_vertical_position(self):
        """Rotate cell so that z is the vertical axis"""
        assert (
            self.hoc_file in self.allowed_hoc_files
        ), "hoc file not in the list of the allowed ones"
        if "L23" in self.hoc_file:
            axis = "x"
            angle = np.pi / 2
        if "L4" in self.hoc_file:
            axis = "x"
            angle = np.pi / 2
        if "L5" in self.hoc_file:
            axis = "y"
            angle = np.pi / 2
        for sec in h.allsec():
            for i in range(sec.n3d()):
                pos = [sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                rot_pos = arbitrary_3d_rotation_along_axis(pos, axis, angle)
                h.pt3dchange(i, *rot_pos, sec.diam3d(i), sec=sec)

    def _move_to_cortical_position(self, cortical_depth):
        assert (
            self.hoc_file in self.allowed_hoc_files
        ), "hoc file not in the list of the allowed ones"
        if cortical_depth == None:
            if "L23" in self.hoc_file:
                self.cortical_depth = 400
            if "L4" in self.hoc_file:
                self.cortical_depth = 790
            if "L5" in self.hoc_file:
                self.cortical_depth = 1150
            print(
                "Warning: Parameter cortical_depth not given when cell was initialized. The parameter was set to %4f um."
                % self.cortical_depth
            )
        else:
            if "L23" in self.hoc_file:
                self.cortical_depth = cortical_depth["L23"]
            if "L4" in self.hoc_file:
                self.cortical_depth = cortical_depth["L4"]
            if "L5" in self.hoc_file:
                self.cortical_depth = cortical_depth["L5"]
        for sec in h.allsec():
            for i in range(sec.n3d()):
                cortex_pos = [sec.x3d(i), sec.y3d(i), sec.z3d(i) - self.cortical_depth]
                h.pt3dchange(i, *cortex_pos, sec.diam3d(i), sec=sec)

    def plot_cell_mapped(
        self,
        df_seg_to_plot_var,
        var_to_plot,
        time,
        norm_axis_view_plane="y",
        scaling=0.1,
        cmap=None,
        ax=None,
        clim=None,
    ):
        from matplotlib.collections import LineCollection
        from matplotlib import pyplot as plt

        if norm_axis_view_plane == "x":
            norm_axis_view_plane = 0
        if norm_axis_view_plane == "y":
            norm_axis_view_plane = 1
        if norm_axis_view_plane == "z":
            norm_axis_view_plane = 2
        plotted_axes = np.delete(["x", "y", "z"], norm_axis_view_plane, axis=0)

        start_end_coords = []
        diams = []
        data = []
        for sec in h.allsec():
            if "Light_source" not in sec.name():
                dsegrange = 1.0 / sec.nseg
                segs_in_sec = [seg for seg in sec]
                for i in range(sec.nseg):
                    start = (
                        sec(i * dsegrange).x_chanrhod,
                        sec(i * dsegrange).y_chanrhod,
                        sec(i * dsegrange).z_chanrhod,
                    )
                    end = (
                        sec((i + 1) * dsegrange).x_chanrhod,
                        sec((i + 1) * dsegrange).y_chanrhod,
                        sec((i + 1) * dsegrange).z_chanrhod,
                    )
                    start_end_coords.append([start, end])
                    diams.append(segs_in_sec[i].diam)
                    data.append(
                        df_seg_to_plot_var.loc[time, str(segs_in_sec[i])][var_to_plot]
                    )

        lines = np.delete(start_end_coords, norm_axis_view_plane, axis=2)
        if not cmap:
            from matplotlib.cm import Blues as cmap
        if not clim:
            clim = [np.min(data), np.max(data)]
        norm_data = (np.array(data) - clim[0]) / (clim[1] - clim[0])

        collection = LineCollection(
            segments=lines, linewidths=np.array(diams) * scaling, colors=cmap(norm_data)
        )
        if ax == None:
            ax = plt.gca()
        ax.add_collection(collection, autolim=True)
        ax.axis("equal")
        ax.set_xlabel("".join([plotted_axes[0], "-axis [um]"]))
        ax.set_ylabel("".join([plotted_axes[1], "-axis [um]"]))
        ax.set_title(
            "".join(
                [
                    self.hoc_file,
                    " - viewing axis: ",
                    ["x", "y", "z"][norm_axis_view_plane],
                ]
            )
        )
        return ax, collection

    def plot_fine(
        self, norm_axis_view_plane="y", scaling=1, color="tab:blue", ax=None
    ):
        from matplotlib.collections import LineCollection
        from matplotlib import pyplot as plt

        if norm_axis_view_plane == "x":
            norm_axis_view_plane = 0
        if norm_axis_view_plane == "y":
            norm_axis_view_plane = 1
        if norm_axis_view_plane == "z":
            norm_axis_view_plane = 2
        plotted_axes = np.delete(["x", "y", "z"], norm_axis_view_plane, axis=0)

        start_end_coords = []
        diams = []
        data = []
        for sec in h.allsec():
            if "Light_source" not in sec.name():
                dsegrange = 1.0 / sec.nseg
                segs_in_sec = [seg for seg in sec]
                for i in range(sec.nseg):
                    start = (
                        sec(i * dsegrange).x_chanrhod,
                        sec(i * dsegrange).y_chanrhod,
                        sec(i * dsegrange).z_chanrhod,
                    )
                    end = (
                        sec((i + 1) * dsegrange).x_chanrhod,
                        sec((i + 1) * dsegrange).y_chanrhod,
                        sec((i + 1) * dsegrange).z_chanrhod,
                    )
                    start_end_coords.append([start, end])
                    diams.append(segs_in_sec[i].diam)

        lines = np.delete(start_end_coords, norm_axis_view_plane, axis=2)

        collection = LineCollection(
            segments=lines, linewidths=np.array(diams) * scaling, colors=color
        )
        if ax == None:
            ax = plt.gca()
        ax.add_collection(collection, autolim=True)
        ax.axis("equal")
        ax.set_xlabel("".join([plotted_axes[0], "-axis [um]"]))
        ax.set_ylabel("".join([plotted_axes[1], "-axis [um]"]))
        ax.set_title(
            "".join(
                [
                    self.hoc_file,
                    " - viewing axis: ",
                    ["x", "y", "z"][norm_axis_view_plane],
                ]
            )
        )
        return ax, collection

    def plot(
        self,
        norm_axis_view_plane="y",
        scaling=0.1,
    ):
        from matplotlib.collections import LineCollection
        from matplotlib import pyplot as plt

        if norm_axis_view_plane == "x":
            norm_axis_view_plane = 0
        if norm_axis_view_plane == "y":
            norm_axis_view_plane = 1
        if norm_axis_view_plane == "z":
            norm_axis_view_plane = 2
        plotted_axes = np.delete(["x", "y", "z"], norm_axis_view_plane, axis=0)

        seg_start_end_coords = []
        seg_diams = []
        for sec in h.allsec():
            sec_start = [sec.x3d(0), sec.y3d(0), sec.z3d(0)]
            sec_intermed_nodes = np.array(
                [[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(1, sec.n3d)]
            )
            sec_intermed_nodes[:-1] += np.diff(sec_intermed_nodes, axis=0) / 2
            sec_end = [
                sec.x3d(sec.nseg + 1),
                sec.y3d(sec.nseg + 1),
                sec.z3d(sec.nseg + 1),
            ]
            sec_start_inter_end_coords = (
                sec_start + list(sec_intermed_nodes[:-1]) + sec_end
            )
            for i in range(sec.nseg):
                seg_start_end_coords.append(
                    [sec_start_inter_end_coords[i], sec_start_inter_end_coords[i + 1]]
                )
            [seg_diams.append(seg.diam) for seg in sec]

        lines = []
        for seg_start, seg_end in seg_start_end_coords:
            lines.append(
                np.delete(
                    [seg_start, seg_end],
                    norm_axis_view_plane,
                    axis=1,
                )
            )

        collection = LineCollection(
            segments=lines, linewidths=np.array(seg_diams) * scaling, colors="tab:blue"
        )
        plt.gca().add_collection(collection, autolim=True)
        plt.axis("equal")
        plt.xlabel("".join([plotted_axes[0], "-axis"]))
        plt.ylabel("".join([plotted_axes[1], "-axis"]))
        plt.title(
            "".join(
                [
                    self.hoc_file,
                    " - viewing axis: ",
                    ["x", "y", "z"][norm_axis_view_plane],
                ]
            )
        )
        return plt.gca()

    def build_tree(self, func,segfunc=False):
        """ func must act on a neuron section
        """
        from numpy import array
        print("-"*100)
        def append_data(sec, xyzdv, parent_id, connections,func,segfunc):
            """ Append data to xyzdv
            """
            if not segfunc: v=func(sec)
            n = int(h.n3d(sec=sec))
            for ii in range(1, n):
                x = h.x3d(ii,sec=sec)
                y = h.y3d(ii,sec=sec)
                z = h.z3d(ii,sec=sec)
                d = h.diam3d(ii,sec=sec)
                if segfunc:
                    if n==1:v=func(sec(0.5))
                    else:v = func(sec(ii/float(n-1)))
                xyzdv.append([x,y,z,d,v])
                child_id = len(xyzdv)-1
                if len(xyzdv)>1:
                    connections.append([child_id, parent_id])
                parent_id = child_id
            return xyzdv, connections

        def append_children_data(parent, parent_id, xyzdv, connections, func, segfunc):
            sref = h.SectionRef(sec=parent)
            if sref.child:
                for child in sref.child:
                    xyzdv, connections = append_data(child, xyzdv, parent_id, connections, func, segfunc)
                    xyzdv, connections = append_children_data(parent = child,
                                                              parent_id = len(xyzdv)-1,
                                                              xyzdv = xyzdv,
                                                              connections = connections,
                                                              func = func,
                                                              segfunc = segfunc)
            return xyzdv, connections

        # Find data and connections
        root_section = h.SectionRef().root
        if segfunc:
            if root_section.nseg==1:
                v = func(root_section(0.5))
            else:
                v = func(root_section(0.0))
        else:
            v=func(root_section)
        xyzdv = [[h.x3d(0,sec=root_section),h.y3d(0,sec=root_section),h.z3d(0,sec=root_section),h.diam3d(0,sec=root_section),v]]
        xyzdv, connections = append_data(root_section, xyzdv, 0, [],func,segfunc)
        xyzdv, connections = append_children_data(root_section,len(xyzdv)-1,xyzdv,connections,func,segfunc)
        return array(xyzdv), array(connections)

    def plot_foutz2012(
            self, func, axes='xz', scaling = 1, segfunc=False, 
            clim=None,cmap=None,lognorm=False,color=None, 
            shift_x=0, shift_y=0, shift_z=0, multiply_x=1, multiply_y=1, multiply_z=1, 
            reverse_draw_order=False, alpha=None): 
        """ plot cell in matplotlib line plot collection
        """
        from numpy import array, linspace
        from matplotlib.collections import LineCollection
        from matplotlib import pyplot
        xyzdv, connections = self.build_tree(func,segfunc)
        deleted_axis = dict(xy=2,xz=1,yz=0,zy=0,zx=1,yx=2)
        xyz = xyzdv[:,:3]
        xyz[:,0] *= multiply_x
        xyz[:,1] *= multiply_y
        xyz[:,2] *= multiply_z
        xyz[:,0] += shift_x
        xyz[:,1] += shift_y
        xyz[:,2] += shift_z
        pts   = np.delete(xyz, deleted_axis[axes],axis=1)
        edges = connections
        diam  = xyzdv[:,3]
        data  = xyzdv[:,4]
        print("DATA RANGE: ",data.min(),data.max())
        # Define colors
        if not clim:
            clim=[data.min(),data.max()]
        if lognorm == True:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=clim[0], vmax=clim[1])
            colors = cmap(norm(data))
        else:
            norm = None
            try:
                a = (data - clim[0])/(clim[1]-clim[0])
                colors = cmap(a)
            except TypeError:
                if color!=None:
                    pass
                else:
                    raise TypeError
        if color:
            colors=color
        if alpha == None:
            alpha = 1    
        # Define line segments
        segments = []
        for edge in edges:
            segments.append([pts[edge[0],:], pts[edge[1],:]])
        # Build Line Collection
        collection = LineCollection(segments = array(segments),
                                    linewidths = diam*scaling,
                                    colors=colors, norm=norm, alpha=alpha)
        if reverse_draw_order:
            # Build Line Collection reversely
            collection = LineCollection(segments = array(segments)[::-1],
                                        linewidths = (diam*scaling)[::-1],
                                        colors=colors[::-1], norm=norm)
        #if cmap:
        #    collection.set_array(data)
        #    collection.set_clim(clim[0], clim[1])
        return collection
