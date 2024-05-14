import os
from neurostim.polarmaps import simple_polar_map
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import Blues
import numpy as np

def plot_active_vs_passive_cell_activation(cell, path):

    cellfiles = [os.path.join(path, '_'.join(['profile',cell, dendstate])) + '.csv' for dendstate in ['active','passive']]

    celldata = []
    for file in cellfiles:
        celldata.append(pd.read_csv(file))
    lps = celldata[0].light_power.unique()
    APCmax = np.max([data.AP_count.max() for data in celldata])

    fig, axs = plt.subplots(ncols=len(lps), nrows=2,
                            subplot_kw={'projection':'polar'},
                            figsize=(14,5)
                           )
    for i, data in enumerate(celldata):
        for j, lp in enumerate(lps):
            ax, mappable = simple_polar_map(
                data=data.loc[data['light_power']==lp],
                plot_col='AP_count',
                ax=axs[i,j],
                vmax=APCmax,
                vmin=0,
                cmap=Blues
            )
            axs[i,j].set_title('lp:{:.2E}'.format(lp))
    for ax in axs.flatten():
        ax.set_yticks([])
        ax.set_xticks([])
    cax = fig.add_axes([0.1,0.1,0.005,0.8])
    plt.colorbar(mappable, cax=cax, orientation='vertical',label='spikes')
    fig.suptitle('     '.join([cell, 'dendrites: top->active; bottom->passive']))
    return fig, ax
