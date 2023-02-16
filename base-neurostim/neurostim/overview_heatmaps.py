import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def merge_2_paramcolumns(df,col1, col2):
    new_col_name = col1 + '_' + col2
    df[new_col_name] = col1 + '=' + df[col1].astype(str)\
      + ' ' + col2 + '=' + df[col2].astype(str)   
    return df, new_col_name

def sorted_4Dto2D_pivot_from_df(
    df, coly1, coly2, colx1, colx2, value_col,
    indexlist_x=None, indexlist_y=None):
    """
    sort and pivot dataframe for fancy_4Din2Dheatmap

    Note: using indexlist_x and _y is highly recommended
          if non-numerical indices are used and if provided
          dataframes have a non-complete index (all possible
          index combinations are explicitly covered in the 
          dataframe.)
    """
    df_merged_x, col_x = merge_2_paramcolumns(df.copy(),colx1, colx2)
    df_merged_xy, col_y = merge_2_paramcolumns(
        df_merged_x, coly1, coly2
    )
    reduced_df_merged_xy = df_merged_xy.loc[:,[col_x,col_y,value_col]]
    pivot = reduced_df_merged_xy.pivot(col_y,col_x,value_col)
    if indexlist_x==None and indexlist_y==None:
        sorted_col_x_idx = reduced_df_merged_xy[col_x].unique()
        sorted_col_y_idx = reduced_df_merged_xy[col_y].unique()
    else:
        sorted_col_x_idx = indexlist_x
        sorted_col_y_idx = indexlist_y
    sorted_pivot = pivot.reindex(
        columns=sorted_col_x_idx
    ).reindex(
        sorted_col_y_idx
    )
    return sorted_pivot

def idxs_where_df_meets_constraint(bool_constrained_df):
    ''' Get number index (row,col) positions of True in dataframe.'''
    # set index to numbers
    bool_constrained_df = bool_constrained_df.reset_index(drop=True) 
    bool_constrained_df.columns = np.arange(bool_constrained_df.shape[1])
    
    listOfPos = list()
    # Get list of columns that contains the value
    seriesObj = bool_constrained_df.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(bool_constrained_df[col][bool_constrained_df[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return np.array(listOfPos)

def fancy_4Din2Dheatmap(
    df,
    pivot,
    colx1, colx2, 
    coly1, coly2, 
    fig, ax, cbar_ax,
    imshow_kws,
    x_label_x_pos, x_label_y_pos,
    y_label_x_pos, y_label_y_pos,
    x1s=None, x2s=None, y1s=None, y2s=None,
    colx1label=None, colx2label=None,
    coly1label=None, coly2label=None,
    cbar_label=None,
    add_xs=False, 
    constraint_idxs_list=None, 
    constrain_color=None, 
    cross_offset=None
):
    def col_label_check(col_label, col):
        if col_label==None:
            return col
        else:
            return col_label
    colx1label = col_label_check(colx1label,colx1) 
    colx2label = col_label_check(colx2label,colx2)
    coly1label = col_label_check(coly1label,coly1) 
    coly2label = col_label_check(coly2label,coly2)
    
    # extract axis of pivoted level 1 and 2 values from columns if needed
    if x1s==None:
        x1s = np.sort(df[colx1].unique())
    if x2s==None:
        x2s = np.sort(df[colx2].unique())
    if y1s==None:
        y1s = np.sort(df[coly1].unique())
    if y2s==None:
        y2s = np.sort(df[coly2].unique())
    # plot imshow and colorbar
    im = ax.imshow(pivot, **imshow_kws)
    fig.colorbar(im, cax=cbar_ax)
    # set ticks and labels on x-axis
    xticks = np.arange(pivot.shape[1])
    upper_x_labels = np.array([str(x2) for x2 in list(x2s)*len(x1s)])
    upper_x_labels[::2] = ['']
    #upper_x_labels = [label if len(label)<4 else label[0] + 'k' for label in upper_x_labels]
    lower_x_labels = np.array(
        [ [str(x1)] + ['']*(len(x2s)-1) for x1 in x1s]
    ).flatten()
    #lower_x_labels = [label if len(label)<4 else label[0] + 'k' for label in lower_x_labels]
    xticklabels = np.char.array(upper_x_labels) + '\n' + np.char.array(lower_x_labels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # set ticks and ticklabels on y-axis
    yticks = np.arange(pivot.shape[0])
    right_y_labels = np.array([str(y2) for y2 in list(y2s)*len(y1s)])
    left_y_labels = np.array(
        [ [str(y1)] + ['']*(len(y2s)-1) for y1 in y1s]
    ).flatten()
    yticklabels = [left_y_labels[i]+ '   ' + right_y_label if len(right_y_label)==2\
                   else left_y_labels[i]+ '  ' + right_y_label\
                   for i,right_y_label in enumerate(right_y_labels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(yticks[0]-0.5,yticks[-1]+0.5)
    # label y axis
    ax.text(s=coly1label+'    '+coly2label, 
            x=y_label_x_pos, 
            y=y_label_y_pos)
    # label x_axis
    ax.text(s=colx1label+'\n'+colx2label, 
            x=x_label_x_pos, 
            y=x_label_y_pos)
    # add vertical and horizontal separation lines for different param regions
    #ax.vlines(x=1,ymin=0,ymax=5)
    ax.vlines(x=np.arange(1,len(x1s))*len(x2s)-0.5, ymin=-0.5, ymax=len(y1s)*len(y2s)+0.5)
    ax.hlines(y=np.arange(1,len(y1s))*len(y2s)-0.5, xmin=-0.5, xmax=len(x1s)*len(x2s)-0.5)   
    # add x's to parametersets, where constraints are met
    if add_xs==True:
        # check if constraints exist:
        if np.array(constraint_idxs_list).size != 0:
            # add constraints if they exist
            for idx, constraint_idxs in enumerate(constraint_idxs_list):
                ax.scatter(constraint_idxs[1] + cross_offset,
                           constraint_idxs[0], 
                           marker='x',color=constrain_color)
    # add colorbar label
    cbar_ax.set_ylabel(cbar_label)
    return fig, ax, cbar_ax
