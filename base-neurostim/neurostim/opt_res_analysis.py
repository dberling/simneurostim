import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

### functions for analysis where APC is fixed to desired value

def prepare_df_for_evaluation(df, defaults_to_reduce):
    """
    - checks for failed parameter searches (added column search_success)
    - turns AP_count afterwards to 0 for all failed paramsets and denotes
    in extra column 'search_success' whether AP search was successful.
    - reduces dataframe with given values for light_model, chanrhod_expression,
      stim_duration (pass in dict via defaults_to_reduce, note: not adaptive to
      other parameters than those 3)
    - creates look-up table for light power used with specific paramset to get
      desired APC
    returns 
    df_reduced, df_light_power_look_up, failedsearch_paramsets
    """
    def return_datasets_where_AP_search_failed(df):
        dfri = df.reset_index()
        datasets = dfri.loc[
            (dfri['radius [um]']==0) & (dfri['angle [rad]']==0) & (dfri['AP_count']=='failed')]
        return datasets
    try:
        assert df.equals(
                df[df.AP_count!='failed']), "data exist where AP count search failed"
        failed_datasets = None
    except AssertionError:
        print('Warning: Data exists where AP count search failed')
        failed_datasets = return_datasets_where_AP_search_failed(df)
    # for param sets where AP count search failed, set AP_count=0 
    # and convert AP_count column dtype to float:
    df['search_success'] = df.groupby(
        ['hoc_file', 'light_model', 'chanrhod_distribution', 'chanrhod_expression',
            'fiber_diameter', 'fiber_NA', 'stim_duration [ms]', 'APC_desired']
         ).apply(lambda df: df['AP_count'][0]!='failed')
    df.loc[df.search_success==False,'AP_count'] = 0
    # set dtype of AP_count columns to float
    df = df.astype(dict(AP_count=float))
    # reduce df columns to variable parameters
    df_ = df.loc[
            :,
            defaults_to_reduce['light_model'],
            :,
            defaults_to_reduce['chanrhod_expression'],
            :,
            :,
            defaults_to_reduce['stim_duration']
            ]
    # create look-up df for light_power
    df_lp = df_.loc[:,:,:,:,:,0,0][['light_power','search_success']]
    df_lp_only_successful_search = df_lp.loc[df_lp.search_success]

    return df_, df_lp_only_successful_search, failed_datasets

def find_x_at_value(xarray, varray, value):
    varray = np.asarray(varray)
    # check if there is a proper stimulation otherwise return x_value = np.nan
    if np.any(varray>value):
        # take first index (if multiple exist) where varray is closest to value
        # !! will cause error in numpy versions > 1.19, resolve as follows then:
        # idx = (np.abs(varray - value)).argmin(keepdims=True)[0]
        idx = (np.abs(varray - value)).argmin()
        # ensure index is the lower idx close to value if array[idx] != value
        while True:
            prev_idx = idx
            idx -= int(varray[idx] < value)
            if idx == prev_idx:
                break
        try:
            assert np.all(varray[idx+1:]<value), "AP count right of found index is larger than value"
        except AssertionError:
            return np.nan
        # interpolate varray to get x according to given value
        x1 = xarray[idx]  
        x2 = xarray[idx+1]
        v1 = varray[idx]
        v2 = varray[idx+1]
        x_value = x1 + (value-v1) * (x2-x1) / (v2-v1)
        return x_value
    else:
        return np.nan

def find_xAPCs(df, longform=False, groupby=None):
    """input: angle averaged df
       output: metadata df with:
       APC = action potential count
       - APC at r=0
       -
       - x where APC = max and max(APC)
       - x where APC = 50% * max and 50%*max(APC)
       - same for 10%
    """
    # add indexes at max(APcount), 50%*max(APcount), 10%*max(APcount)
    df = df.join(
        pd.DataFrame(
        df[['AP_count', 'APC_desired_aux']].groupby(groupby).apply(
        lambda df: np.argmax(df['AP_count'])), columns=['i_APCmax'])
    )
    df = df.join(
        pd.DataFrame(
        df[['AP_count', 'APC_desired_aux','i_APCmax']].groupby(groupby).apply(
        lambda df: find_x_at_value(
            xarray=df.index.get_level_values('radius [um]')[df['i_APCmax'][0]:],
            varray=df['AP_count'][df['i_APCmax'][0]:], 
            value=df['APC_desired_aux'][0] * 0.5)),
        columns=['x_APC50'])
    )
    df = df.join(pd.DataFrame(
        df[['AP_count', 'APC_desired_aux','i_APCmax','x_APC50']].groupby(groupby).apply(
        lambda df: find_x_at_value(
            xarray=df.index.get_level_values('radius [um]')[df['i_APCmax'][0]:],
            varray=df['AP_count'][df['i_APCmax'][0]:], 
            value=df['APC_desired_aux'][0] * 0.1)),
        columns=['x_APC10'])
    )
    # add APcount at 0
    df = df.join(pd.DataFrame(
        df[['AP_count', 'APC_desired_aux','i_APCmax','x_APC50','x_APC10']].groupby(groupby).apply(
        lambda df: df['AP_count'][0]), columns=['APC_x0'])
    )
    # add radius ("x") at max(APcount), 50%*max(APcount), 10%*max(APcount)
    df = df.join(pd.DataFrame(
        df[['AP_count', 'APC_desired_aux','i_APCmax','x_APC50','x_APC10','APC_x0','APCmax']].groupby(groupby).apply(
        lambda df: df.index.get_level_values('radius [um]')[df['i_APCmax'][0]]), columns=['x_APCmax']
    ), how='outer')
    # remove radii and respective AP counts from dataframe to keep only metadata
    df = df[['i_APCmax', 'APC_desired_aux','x_APC50','x_APC10', 'APC_x0','APCmax','x_APCmax']].groupby(groupby).apply(
        lambda df: df.iloc[0])
    if longform:
        # convert from wide into long form
        df_melted = pd.melt(df[['x_APCmax', 'x_APC50', 'x_APC10']].reset_index(),
                id_vars=groupby,
                value_vars=['x_APCmax', 'x_APC50', 'x_APC10'],
               value_name='radius [um]').set_index(groupby)
        return df_melted
    return df

def eval_spatial_resolution(df, APC_desired_tolerance=0.1):
    """
    - Check if stimulation achieved APC_desired
    - evaluate position (distance from center) of APCmax, APC50, APC10
    - take the farthest distance from center where APC is X% from maximum
      to find position (x) where APC is 50% or 10% (APC50 / APC10 respectively)
    
    assume input df:
    hoc_file/chanrhod_distribution/fiber_diameter/fiber_NA/APC_desired/radius [um]/angle [rad]

    returns df_resolutions, non_monotonic_failed_paramsets
    """
    # average over angle to get radial APC profiles
    df = pd.DataFrame(
        df[['firint_rate [Hz]', 'AP_count']].groupby([
            'hoc_file', 'chanrhod_distribution',
            'fiber_diameter','fiber_NA','APC_desired','radius [um]']
        ).mean()
    )
    # check if stimulation achieved APC_desired
    df['APCmax'] = pd.DataFrame(
        df['AP_count'].groupby([
            'hoc_file', 'chanrhod_distribution',
            'fiber_diameter','fiber_NA','APC_desired']
        ).max()
    )
    # set AP_count to 0 for datasets where APC_desired is not reached (incl. tolerance)
    df['APC_desired_aux'] = df.index.get_level_values('APC_desired')
    df['stim_success'] =(df.APC_desired_aux * (1 - APC_desired_tolerance)) < df['APCmax']
    df.loc[df.stim_success==False, 'AP_count'] = 0
    # calculate positions of APCmax, APC50, APC10 from datasets where stimulation successful
    #return df.loc[df.stim_success]
    df_res = find_xAPCs(
        df, longform=False, 
        groupby=[
        'hoc_file', 'chanrhod_distribution', 'fiber_diameter',
        'fiber_NA', 'APC_desired']
    )
    failed_paramsets = df.loc[df.stim_success==False]
    return df_res, failed_paramsets


### functions for analysis where the response for different light powers is simulated 
### and the spatial resolution shall be evaluated and plotted as a lineplot

def avrg_angles(df, groupby=None):
    if groupby == None:
        groupby = [
            'hoc_file', 'light_model', 'chanrhod_distribution', 'chanrhod_expression',
            'fiber_diameter','fiber_NA','stim_duration [ms]','light_power','radius [um]'
        ]
    avrg_angles = pd.DataFrame(
        df[['firint_rate [Hz]', 'AP_count']].groupby(groupby).mean()
    )
    return avrg_angles


def find_xAPCs_over_light_pwrs(df, longform=False, groupby=None, rad_label='radius [um]'):
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
        groupby = ['hoc_file', 'light_model','chanrhod_distribution','chanrhod_expression',
                   'fiber_diameter', 'fiber_NA', 'stim_duration [ms]','light_power']
    # add indexes at max(APcount), 50%*max(APcount), 10%*max(APcount)
    df = df.join(
        pd.DataFrame(
        df[['AP_count']].groupby(groupby).apply(
        lambda df: np.argmax(df['AP_count'])), columns=['i_APCmax'])
    )
    df = df.join(
        pd.DataFrame(
        df[['AP_count', 'i_APCmax']].groupby(groupby).apply(
        lambda df: np.max(df['AP_count'])), columns=['APCmax'])
    )
    df = df.join(
        pd.DataFrame(
        df[['AP_count', 'i_APCmax', 'APCmax']].groupby(groupby).apply(
        lambda df: find_x_at_value(
            xarray=df.index.get_level_values(rad_label)[df['i_APCmax'][0]:],
            varray=df['AP_count'][df['i_APCmax'][0]:],
            value=df['APCmax'][0] * 0.5)),
        columns=['x_APC50'])
    )
    df = df.join(pd.DataFrame(
        df[['AP_count', 'APCmax','i_APCmax','x_APC50']].groupby(groupby).apply(
        lambda df: find_x_at_value(
            xarray=df.index.get_level_values(rad_label)[df['i_APCmax'][0]:],
            varray=df['AP_count'][df['i_APCmax'][0]:],
            value=df['APCmax'][0] * 0.1)),
        columns=['x_APC10'])
    )
    # add APcount at 0
    df = df.join(pd.DataFrame(
        df[['AP_count', 'APCmax','i_APCmax','x_APC50','x_APC10']].groupby(groupby).apply(
        lambda df: df['AP_count'][0]), columns=['APC_x0'])
    )
    # add radius ("x") at max(APcount), 50%*max(APcount), 10%*max(APcount)
    df = df.join(pd.DataFrame(
        df[['AP_count', 'APCmax','i_APCmax','x_APC50','x_APC10','APC_x0','APCmax']].groupby(groupby).apply(
        lambda df: df.index.get_level_values(rad_label)[df['i_APCmax'][0]]), columns=['x_APCmax']
    ), how='outer')
    # remove radii and respective AP counts from dataframe to keep only metadata
    df = df[['i_APCmax','x_APC50','x_APC10', 'APC_x0','APCmax','x_APCmax']].groupby(groupby).apply(
        lambda df: df.iloc[0])
    if longform:
        # convert from wide into long form
        df_melted = pd.melt(df[['x_APCmax', 'x_APC50', 'x_APC10']].reset_index(),
                id_vars=groupby,
                value_vars=['x_APCmax', 'x_APC50', 'x_APC10'],
               value_name=rad_label).set_index(groupby)
        return df_melted
    return df

def lineplot_x_APs_and_APC(paramsetdf,paramsetdf_long, ax=None, c1='tab:blue', c2='tab:red'):
    import seaborn as sns
    if ax==None:
        fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.lineplot(data=paramsetdf_long, x= 'light_power', y='radius [um]',
            style='variable', color=c1, ax=ax)
    ax.set_xscale('log')
    ax.set_ylabel('distance from central position [um]')
    ax2 = ax.twinx()
    ax2.plot(paramsetdf.reset_index().light_power, paramsetdf.APCmax,color=c2)
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

#### IMPROVED Feb 24
def find_xAPC50(df):
    """
    improved version to calc response radius
    """
    df = df.reset_index()
    if df.AP_count.sum() < 1:
        return np.nan
    if 'radius [um]' in df.columns:
        df = df.rename(columns={'radius [um]': 'radius'})
    interpolation = interp1d(df.radius.values, df.AP_count.values)
    # radius at peak response
    x_APCmax = df[df.AP_count==df.AP_count.max()].radius.values[0]
    rmax = df.radius.max()
    radii_interpolated = np.linspace(0,rmax,rmax)
    APC_interpolated = interpolation(radii_interpolated)
    # find largest radius at which AP count is half-max
    tolerance = np.sqrt(df.AP_count.max()) * 0.05 + 0.01
    x_APC50_outmost = np.max(radii_interpolated[np.abs(APC_interpolated-df.AP_count.max()/2)<tolerance])
    return x_APC50_outmost

def improved_find_xAPC50_over_lps(df, groupby=None, apply_to='AP_count'):
    """
    Find response radius (APC50) and peak response (APCmax).
    - avrg over angle
    - groupby (w/o radius)
    - calc rr and pr
    """
    if apply_to != 'AP_count':
        df = df.rename(columns={apply_to: 'AP_count'})
    if groupby == None:
        groupby = [
            'hoc_file', 'light_model', 'chanrhod_distribution', 'chanrhod_expression',
            'fiber_diameter','fiber_NA','stim_duration [ms]','light_power'
        ]
    if 'radius' in df.reset_index().columns:
        rad_label = 'radius'
    elif 'radius [um]' in df.reset_index().columns:
        rad_label = 'radius [um]'
    else:
        raise ValueError("Cannot detect radius column in dataframe df.")
    # make radius column dtype int
    df['radius [um]'] = df['radius [um]'].astype('int64')
    # avrg angles
    df_angavrg = df.groupby(groupby+[rad_label]).mean()
    # calc peak response
    APCmax = df_angavrg.groupby(groupby).max()
    # calc response radius
    xAPC50 = df_angavrg.groupby(groupby).apply(lambda x:find_xAPC50(x.reset_index()))
    
    spres = pd.DataFrame(APCmax).join(pd.DataFrame(xAPC50, columns=['xAPC50'])).rename(columns={apply_to:'APCmax'})
    return spres
