from neuron import h
import numpy as np

"""
model:
    load hoc
    solve cortical depth
    rotation
    insertion of biophysics including ChR
"""

class Cell:
    def __init__(
        self,
        model,
        ChR_soma_density,
        ChR_distribution,
    ):
        """
        Parameters:
        -----------

        model: object(, optional: h. object)
            model object(s) as returned from functions defined in models.py
        ChR_soma_density: float
            ChR density in soma in 1/cm2, Foutz et al use 13e9 /cm2
        ChR_distribution: {'uniform', 'shemesh_fig1m_untrgtd', shemesh_fig1n_trgtd'}
            Type of distribution to distirbute the ChR channels over the morphology
        """
        self.model = model[0]
        self.h_obj = model[1]
        self._distribute_ChR_density(
            ChR_soma_density=ChR_soma_density,
            distribution=ChR_distribution
        )
        self._assign_pos_chanrhod()

    def _assign_pos_chanrhod(self):
        """Assign x, y, and z chanrhod to neuron section"""
        for sec in list(h.allsec()):
            if h.ismembrane("chanrhod", sec=sec):
                try:
                    n = sec.n3d()
                    x = h.Vector(n)
                    y = h.Vector(n)
                    z = h.Vector(n)
                    length = h.Vector(n)
                    for i in range(n):
                        x.x[i] = sec.x3d(i)
                        y.x[i] = sec.y3d(i)
                        z.x[i] = sec.z3d(i)
                        length.x[i] = sec.arc3d(i)
                    length.div(length.x[n - 1])
                    r = h.Vector(sec.nseg + 2)
                    r.indgen(1.0 / sec.nseg)
                    r.sub(1.0 / (2.0 * sec.nseg))
                    r.x[0] = 0
                    r.x[sec.nseg + 1] = 1
                    x_int = h.Vector(sec.nseg + 2)
                    y_int = h.Vector(sec.nseg + 2)
                    z_int = h.Vector(sec.nseg + 2)
                    x_int.interpolate(r, length, x)
                    y_int.interpolate(r, length, y)
                    z_int.interpolate(r, length, z)
                    for i in range(1, sec.nseg + 1):
                        xr = r.x[i]
                        sec(xr).x_chanrhod = x_int.x[i]
                        sec(xr).y_chanrhod = y_int.x[i]
                        sec(xr).z_chanrhod = z_int.x[i]
                except IndexError:
                    print(sec)
                    pass

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
                    distance_from_soma_center = h.distance(self.model.soma_sec(0.5), seg)  # um
                    seg.channel_density_chanrhod = ChR_soma_density * self._get_ChR_expression_level(
                        distance_from_soma_center, distribution=distribution
                    )
        return None

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
        return collection
