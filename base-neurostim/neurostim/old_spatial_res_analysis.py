import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def avrg_angles(df):
    avrg_angles = pd.DataFrame(
        df[['firint_rate [Hz]', 'AP_count']].groupby([
            'hoc_file', 'light_model', 'chanrhod_distribution', 'chanrhod_expression',
            'fiber_diameter','fiber_NA','stim_duration [ms]','light_power','radius [um]']
        ).mean()
    )
    return avrg_angles

def find_APmax_50_10(df, longform=False, groupby=None):
    """input: angle averaged df
       output: metadata df with:
       APC = action potential count
       - APC at r=0
       -
       - x where APC = max and max(APC)
       - x where APC = 50% * max and 50%*max(APC)
       - same for 10%
    """
    if groupby==None:
        groupby = ['hoc_file', 'light_model', 'chanrhod_distribution', 'chanrhod_expression',
                   'fiber_diameter', 'fiber_NA', 'stim_duration [ms]', 'light_power']
        print('use default groupby:')
        print(groupby)

    # add indexes at max(APcount), 50%*max(APcount), 10%*max(APcount)
    df = df.join(
        pd.DataFrame(
        df[['AP_count']].groupby(groupby).apply(
        lambda df: np.argmax(df['AP_count'])), columns=['i_AP_max'])
    )
    df = df.join(
        pd.DataFrame(
        df[['AP_count','i_AP_max']].groupby(groupby).apply(
        lambda df: df['i_AP_max'][0] + find_nearest_idx(df['AP_count'][df['i_AP_max'][0]:], df['AP_count'][df['i_AP_max'][0]] * 0.5)),
        columns=['i_AP_50'])
    )
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50']].groupby(groupby).apply(
        lambda df: df['i_AP_50'][0] + find_nearest_idx(df['AP_count'][df['i_AP_50'][0]:], df['AP_count'][df['i_AP_max'][0]] * 0.1)),
        columns=['i_AP_10'])
    )
    # add APcount at 0
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10']].groupby(groupby).apply(
        lambda df: df['AP_count'][0]), columns=['AP_0'])
    )
    # add APcount and radius ("x") at max(APcount), 50%*max(APcount), 10%*max(APcount)
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10','AP_0']].groupby(groupby).apply(
        lambda df: df['AP_count'][df['i_AP_max'][0]]), columns=['AP_max']
    ), how='outer')
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10','AP_0','AP_max']].groupby(groupby).apply(
        lambda df: df.index.get_level_values('radius [um]')[df['i_AP_max'][0]]), columns=['x_AP_max']
    ), how='outer')
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10','AP_0','AP_max','x_AP_max']].groupby(
            groupby).apply(
        lambda df: df['AP_count'][df['i_AP_50'][0]]), columns=['AP_50']
    ), how='outer')
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10','AP_0','AP_max','x_AP_max','AP_50']].groupby(
            groupby).apply(
        lambda df: df.index.get_level_values('radius [um]')[df['i_AP_50'][0]]), columns=['x_AP_50']
    ), how='outer')
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10','AP_0','AP_max','x_AP_max','AP_50','x_AP_50']].groupby(groupby).apply(
        lambda df: df['AP_count'][df['i_AP_10'][0]]), columns=['AP_10']
    ), how='outer')
    df = df.join(pd.DataFrame(
        df[['AP_count','i_AP_max','i_AP_50','i_AP_10', 'AP_0','AP_max','x_AP_max','AP_50','x_AP_50','AP_10']].groupby(groupby).apply(
        lambda df: df.index.get_level_values('radius [um]')[df['i_AP_10'][0]]), columns=['x_AP_10']
    ), how='outer')
    # remove radii and respective AP counts from dataframe to keep only metadata
    df = df[['i_AP_max','i_AP_50','i_AP_10', 'AP_0','AP_max','x_AP_max','AP_50','x_AP_50','AP_10','x_AP_10']].groupby(groupby).apply(
        lambda df: df.iloc[0])
    if longform:
        # convert from wide into long form
        df_melted = pd.melt(df[['x_AP_max', 'x_AP_50', 'x_AP_10']].reset_index(),
                id_vars=groupby,
                value_vars=['x_AP_max', 'x_AP_50', 'x_AP_10'],
               value_name='radius [um]').set_index(groupby)
        return df_melted
    return df

def lineplot_x_APs_and_APC(paramsetdf,paramsetdf_long, ax=None, c1='tab:blue', c2='tab:red'):
    if ax==None:
        fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.lineplot(data=paramsetdf_long, x= 'light_power', y='radius [um]', 
            style='variable', color=c1, ax=ax)
    ax.set_xscale('log')
    ax.set_ylabel('distance from central position [um]')
    ax2 = ax.twinx()
    ax2.plot(paramsetdf.reset_index().light_power, paramsetdf.AP_max,color=c2)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel('AP count (---)')
    ax2.spines['left'].set_color(c1)
    ax.tick_params(axis='y', colors=c1)
    ax2.spines['right'].set_color(c2)
    ax2.tick_params(axis='y', colors=c2)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.set_ylabel('distance from center [um]', color=c1)
    ax2.set_ylabel('AP count', color=c2)
    ax.set_ylim(0,400)
    ax2.set_ylim(0,30)
    return ax, ax2
