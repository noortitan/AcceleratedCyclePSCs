# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:25:21 2023

@author: Titan Hartono (titan.hartono@helmholtz-berlin.de)
"""

#%% Functions & fixed parameters

import matplotlib.pyplot as plt
# import sys
import os
import numpy as np
import pandas as pd
# from pandas import DataFrame, read_csv
# from IPython.display import display_html
import seaborn as sns

# import plotly.express as px
# import plotly.graph_objs as go
# import plotly.io as pio

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import median_absolute_error
import scipy

from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.integrate import simps
from scipy import integrate
from scipy.stats import linregress

###############################################################################
# PARAMETERS/ SET UP
###############################################################################

# The cycle hours
hour_cycle_25C = 12
gap_25C = 2 # At least, the gap is how many hours?
temperature_25C = 25

hour_cycle_35C = 6
gap_35C = 1.5 # At least, the gap is how many hours?
temperature_35C = 35

hour_cycle_45C = 3
gap_45C = 1.5 # At least, the gap is how many hours?
temperature_45C = 45

hour_cycle_55C = 1.5
gap_55C = 1 # At least, the gap is how many hours?
temperature_55C = 55

parameter_list = ['VmppArea_perCycle','VmppAreaLoss_perCycle',
                  'PCEmppArea_perCycle','PCEmppAreaLoss_perCycle',
                  'mean_Vmpp','max_Vmpp',
                  'mean_PCEmpp','max_PCEmpp']

parameter_loss_list = ['VmppArea_perCycle','VmppAreaLoss_perCycle',
                       'PCEmppArea_perCycle','PCEmppAreaLoss_perCycle']

parameter_mean_list = ['mean_VmppArea_perCycle','mean_PCEmppArea_perCycle',
                       'mean_VmppAreaLoss_perCycle','mean_PCEmppAreaLoss_perCycle']

parameter_std_list = ['std_VmppArea_perCycle','std_PCEmppArea_perCycle',
                      'std_VmppAreaLoss_perCycle','std_PCEmppAreaLoss_perCycle']

parameter_ratio_list = ['PCE_mpp_ratio','V_mpp_ratio']

batchname_list = ['Ulas','Zahra-Batch1', 'Zahra-Batch2']

# # Get the directory of the current script or file
# current_file_directory = os.path.dirname(os.path.abspath(__file__))
current_file_directory = 'C:/Users/Titan/Dropbox (Personal)/backup_MIT/HZB/AcceleratedCycle/'

# Set the current working directory to the file's directory
os.chdir(current_file_directory)

# Turn off warning
pd.options.mode.chained_assignment = None # default='warn'

###############################################################################
# FUNCTIONS: LOAD DATA 
###############################################################################

# Function to load all pixel data into dataframe
def load_data_into_df(hour_cycle, gap, temperature, path):
    device_name = []
    pixel = []
    df_list = []

    # Go through each file on the list
    for filename in os.listdir(os.getcwd()+path):

        # Split the name
        name, extension = os.path.splitext(filename)
        name_split = name.split("_")

        # Look at if the name is Ulas or Zahra, and append name accordingly
        if name_split[6] == 'Zahra':
            device_name.append(name_split[6]+"-"+name_split[7]) # Append the name to the list
        else:
            device_name.append(name_split[7]) # Append the name to the list

        # Append the pixel name to the list 
        pixel.append(name_split[4])

        # Open the file, and create a dataframe based on that
        with open(os.getcwd()+path+filename) as f:

            # Load data
            array = np.loadtxt(f)#(os.getcwd()+path+filename)

            # Load the array as dataframe, calculate time in hours, and drop NaN rows
            df_array = pd.DataFrame(array,columns = ['time_s','PCE_mpp','V_mpp','J_mpp'])
            df_array['time_h'] = df_array['time_s']/3600
            df_array = df_array.dropna().reset_index(drop=True)

            # Perform operation on df_array

            # Introduce counter for num
            time0 = 0
            cycle = 1

            # Adding a new column for the cycle
            df_array['cycle'] = 0

            # Go through each row to calculate time delta and cycle number
            for index,row in df_array.iterrows():

                # Calculate time_delta
                if index == 0:
                    time_delta = df_array['time_h'][index]-time0
                else:
                    time_delta = df_array['time_h'][index]-df_array['time_h'][index-1]

                # Calculate cycle #
                if time_delta < gap:
                    df_array['cycle'][index] = cycle
                else:
                    df_array['cycle'][index] = cycle+1
                    cycle+=1

            # Calculate 'collapsed time', reset t_h to 0 for the beginning of each cycle
            df_array['time_h_collapsed'] = df_array['time_h']-(df_array['cycle']-1)*hour_cycle*2

            df_list.append(df_array)


    # Create a dataframe with this
    df_all = pd.DataFrame(list(zip(device_name, pixel,df_list)),columns =['device_name', 'pixel','MPPT'])

    # Add the temperature column to the df
    df_all['temperature'] = temperature
    df_all['hour_cycle'] = hour_cycle
    
    return df_all

# Loading the 35C from python processed using load_mpp_hysprint

# Function to load all pixel data into dataframe
def load_data_from_python_into_df(hour_cycle, gap, temperature, path, cell_params, limit_h = None):
    device_name = []
    pixel = []
    df_list = [] 

    # Go through each file on the list
    for filename in os.listdir(os.getcwd()+path):

        # Split the name
        name, extension = os.path.splitext(filename)
        name_split = name.split("_")
        
        # Append device name and pixel
        pixel.append('dev'+str(int(name_split[2])+1)+'pix'+str(int(name_split[4])+1))
        device_name.append(cell_params[int(name_split[2])][1])

        # Load data csv
        array = pd.read_csv(os.getcwd()+path+filename)
        
        # Only select certain columns
        df_array = array[['Duration_s','MPPT_EFF','MPPT_V','MPPT_J','Duration_h']]
        
        # Change column names so they are the same
        new_column_names = {'Duration_s': 'time_s',
                            'MPPT_EFF': 'PCE_mpp',
                            'MPPT_V': 'V_mpp',
                            'MPPT_J': 'J_mpp',
                            'Duration_h': 'time_h'}
        
        df_array.rename(columns=new_column_names, inplace=True)
        
        # If limit_h is not None, you basically limit the x axis of the MPPT
        if limit_h is not None:
            
            df_array=df_array[df_array['time_h'] < limit_h]
            
        
        # Make sure it's positive value? abs of MPPT_V
        df_array['V_mpp'] = df_array['V_mpp'].abs()
        
        # Perform operation on df_array
        
        # Introduce counter for num
        time0 = 0
        cycle = 1
        
        # Adding a new column for the cycle
        df_array['cycle'] = 0
        
        # Go through each row to calculate time delta and cycle number
        for index,row in df_array.iterrows():
            
            # Calculate time_delta
            if index == 0:
                time_delta = df_array['time_h'][index]-time0
            else:
                time_delta = df_array['time_h'][index]-df_array['time_h'][index-1]
            
            # Calculate cycle #
            if time_delta < gap:
                df_array['cycle'][index] = cycle
            else:
                df_array['cycle'][index] = cycle+1
                cycle+=1
        
        # Calculate 'collapsed time', reset t_h to 0 for the beginning of each cycle
        df_array['time_h_collapsed'] = df_array['time_h']-(df_array['cycle']-1)*hour_cycle*2
        
        df_list.append(df_array)


    # Create a dataframe with this
    df_all = pd.DataFrame(list(zip(device_name, pixel,df_list)),columns =['device_name', 'pixel','MPPT'])

    # Add the temperature column to the df
    df_all['temperature'] = temperature
    df_all['hour_cycle'] = hour_cycle
    
    return df_all

###############################################################################
# FUNCTIONS: CALCULATE AREA
###############################################################################

def calculate_area_perCycle(df_all):#, h_cycle):#, type_AccCyc):
    
    # List for the area df
    area_df = []
    
    # Go to each row in the array
    for index,row in df_all.iterrows():
        
        # Select df
        df_selected = df_all['MPPT'][index]
        h_cycle = df_all['hour_cycle'][index]
    
        # Unique cycles
        unique_cycle = df_selected['cycle'].unique()

        # Integrate using trapz method from scipy
        VmppArea_perCycle = df_selected.groupby(df_selected.cycle).apply(lambda g: integrate.trapz(g.V_mpp, x=g.time_h_collapsed))
        PCEmppArea_perCycle = df_selected.groupby(df_selected.cycle).apply(lambda g: integrate.trapz(g.PCE_mpp, x=g.time_h_collapsed))

        # Convert to series
        area_perCycle = ((VmppArea_perCycle.to_frame()).rename(columns={0:'VmppArea_perCycle'})).reset_index()
        area_perCycle['PCEmppArea_perCycle'] = PCEmppArea_perCycle.tolist()
        #((PCEmppArea_perCycle.to_frame()).rename(columns={0:'PCEmppArea_perCycle'})).reset_index()

        # Multiply with 36000s/h and P_in = 100 mW/cm2, and divide by 1000 (MILI watt)
        area_perCycle['PCEmppArea_perCycle_perCellArea']=area_perCycle['PCEmppArea_perCycle']*3600*100/1000

        # Add another column
        area_perCycle['cycle_h'] = (area_perCycle['cycle']-1)*h_cycle*2

        # Now, let's calculate what the max, average for each Vmpp and PCEmpp
        max_Vmpp_list = []
        max_PCEmpp_list = []
        mean_Vmpp_list = []
        mean_PCEmpp_list = []

        for i in unique_cycle:
            # Select for a specific cycle
            df_cycle = df_selected[df_selected['cycle']==i]

            # Append with mean and max values
            mean_Vmpp_list.append(df_cycle['V_mpp'].mean())
            mean_PCEmpp_list.append(df_cycle['PCE_mpp'].mean())
            max_Vmpp_list.append((df_cycle.nlargest(3, 'V_mpp'))['V_mpp'].mean()) # Average of top 5 values
            max_PCEmpp_list.append((df_cycle.nlargest(3, 'PCE_mpp'))['PCE_mpp'].mean())# Average of top 5 values

        # Convert the list into df column
        area_perCycle['mean_Vmpp'] = mean_Vmpp_list
        area_perCycle['mean_PCEmpp'] = mean_PCEmpp_list
        area_perCycle['max_Vmpp'] = max_Vmpp_list
        area_perCycle['max_PCEmpp'] = max_PCEmpp_list

        # Calculate the area loss
        area_perCycle['VmppAreaLoss_perCycle'] = (area_perCycle['max_Vmpp']*h_cycle)-area_perCycle['VmppArea_perCycle']
        area_perCycle['PCEmppAreaLoss_perCycle'] = (area_perCycle['max_PCEmpp']*h_cycle)-area_perCycle['PCEmppArea_perCycle']

        # Drop the last row
        area_perCycle = area_perCycle.iloc[:-1 , :]
        
        area_df.append(area_perCycle)
    
    df_all['area_perCycle'] = area_df
    
    return df_all

# Function to calculate stats of the area 
def calculate_area_perCycle_stats(df_all_area):
    # Unique device name_int
    unique_device = df_all_area['device_name'].unique()

    list_stats=[]
    list_temperature = []
    list_hour_cycle = []

    for device_name_int in unique_device:

        # Select specific device_name
        df_all_area_int =  df_all_area[df_all_area['device_name']==device_name_int].reset_index(drop=True)
        len_df = len(df_all_area_int)

        # Creating empty list
        list_VmppArea_perCycle = []
        list_PCEmppArea_perCycle = []
        list_VmppAreaLoss_perCycle = []
        list_PCEmppAreaLoss_perCycle = []

        # Go through all the rows for specific device_name
        for index,row in df_all_area_int.iterrows():

            list_VmppArea_perCycle.append(df_all_area_int['area_perCycle'][index]['VmppArea_perCycle'].values.tolist())
            list_PCEmppArea_perCycle.append(df_all_area_int['area_perCycle'][index]['PCEmppArea_perCycle'].values.tolist())
            list_VmppAreaLoss_perCycle.append(df_all_area_int['area_perCycle'][index]['VmppAreaLoss_perCycle'].values.tolist())
            list_PCEmppAreaLoss_perCycle.append(df_all_area_int['area_perCycle'][index]['PCEmppAreaLoss_perCycle'].values.tolist())

        # Convert list to numpy arrays
        np_VmppArea_perCycle = np.array(list_VmppArea_perCycle)
        np_PCEmppArea_perCycle = np.array(list_PCEmppArea_perCycle) 
        np_VmppAreaLoss_perCycle = np.array(list_VmppAreaLoss_perCycle)
        np_PCEmppAreaLoss_perCycle = np.array(list_PCEmppAreaLoss_perCycle) 

        # Calculate statistics (mean and standard. deviation)
        mean_VmppArea_perCycle = np.mean(np_VmppArea_perCycle,axis=0)
        mean_PCEmppArea_perCycle = np.mean(np_PCEmppArea_perCycle,axis=0)
        mean_VmppAreaLoss_perCycle = np.mean(np_VmppAreaLoss_perCycle,axis=0)
        mean_PCEmppAreaLoss_perCycle = np.mean(np_PCEmppAreaLoss_perCycle,axis=0)

        std_VmppArea_perCycle = np.std(np_VmppArea_perCycle,axis=0)
        std_PCEmppArea_perCycle = np.std(np_PCEmppArea_perCycle,axis=0)
        std_VmppAreaLoss_perCycle = np.std(np_VmppAreaLoss_perCycle,axis=0)
        std_PCEmppAreaLoss_perCycle = np.std(np_PCEmppAreaLoss_perCycle,axis=0)

        # Creating a df for the stats
        df_stats_int = df_all_area_int['area_perCycle'][0][['cycle','cycle_h']]

        df_stats_int['mean_VmppArea_perCycle'] = mean_VmppArea_perCycle
        df_stats_int['mean_PCEmppArea_perCycle'] = mean_PCEmppArea_perCycle
        df_stats_int['mean_VmppAreaLoss_perCycle'] = mean_VmppAreaLoss_perCycle
        df_stats_int['mean_PCEmppAreaLoss_perCycle'] = mean_PCEmppAreaLoss_perCycle

        df_stats_int['std_VmppArea_perCycle'] = std_VmppArea_perCycle
        df_stats_int['std_PCEmppArea_perCycle'] = std_PCEmppArea_perCycle
        df_stats_int['std_VmppAreaLoss_perCycle'] = std_VmppAreaLoss_perCycle
        df_stats_int['std_PCEmppAreaLoss_perCycle'] = std_PCEmppAreaLoss_perCycle

        # Append the df_stats into list_stats
        list_stats.append(df_stats_int)
        list_temperature.append(df_all_area_int['temperature'][0])
        list_hour_cycle.append(df_all_area_int['hour_cycle'][0])

    df_stats = pd.DataFrame()
    df_stats['device_name'] = unique_device
    df_stats['temperature'] = list_temperature
    df_stats['hour_cycle'] = list_hour_cycle
    df_stats['area_perCycle_stats'] = list_stats
    
    return df_stats

###############################################################################
# FUNCTIONS: CALCULATE PCEMPP/ VMPP O VS 1000H + PLOT RATIO
###############################################################################

# Function to extract df PCE and Vmpp at time = 0 and 1000h
def calculate_Vmpp_PCEmpp_0_1000(df):
    # Input: df_all_25C, df_all_45C, df_all_55C
    
    # Empty lists for appending extracted values
    list_PCEmpp_0 = []
    list_Vmpp_0 = []
    list_PCEmpp_1000 = []
    list_Vmpp_1000 = []
    list_t1000 = []

    # Go through every row
    for index,row in df.iterrows():
        
        # Select df with particular index
        selected_df=(df['MPPT'][index])
        
        # Select df with time_h <1000
        less_1000 = selected_df.loc[(selected_df['time_h'] < 1000)]
        larger_1000 = selected_df.loc[(selected_df['time_h'] >= 1000)]
        
        # See if this reaches 1000th hour        
        if larger_1000.empty == False:
            # Append the 1000th hour V_mpp and PCE_mpp
            list_Vmpp_1000.append((less_1000.iloc[-1])['V_mpp'])
            list_PCEmpp_1000.append((less_1000.iloc[-1])['PCE_mpp'])
            list_t1000.append((less_1000.iloc[-1])['time_h'])
            
        else:
            # Append the 1000th hour V_mpp and PCE_mpp
            list_Vmpp_1000.append((selected_df.iloc[-1])['V_mpp'])
            list_PCEmpp_1000.append((selected_df.iloc[-1])['PCE_mpp'])
            list_t1000.append((selected_df.iloc[-1])['time_h'])
        
        # Append the 0th hour V_mpp and PCE_mpp
        list_PCEmpp_0.append((df['MPPT'][0])['PCE_mpp'][0])
        list_Vmpp_0.append((df['MPPT'][0])['V_mpp'][0])

    # Create the dataframe
    df_0_1000 = pd.DataFrame()
    df_0_1000[['device_name','pixel','temperature','hour_cycle']]= df[['device_name','pixel','temperature','hour_cycle']]
    
    # Add the lists into the dataframe
    df_0_1000['V_mpp_1000'] = list_Vmpp_1000
    df_0_1000['PCE_mpp_1000'] = list_PCEmpp_1000
    df_0_1000['V_mpp_0'] = list_Vmpp_0
    df_0_1000['PCE_mpp_0'] = list_PCEmpp_0
    df_0_1000['t1000_h'] = list_t1000
    
    df_0_1000['PCE_mpp_ratio'] = df_0_1000['PCE_mpp_1000']/df_0_1000['PCE_mpp_0']
    df_0_1000['V_mpp_ratio'] = df_0_1000['V_mpp_1000']/df_0_1000['V_mpp_0']
    
    return df_0_1000

# Plot comparison ratio for 0 vs 1000 h
def plot_ratio_Vmpp_PCEmpp_0_1000(df_0_1000_25C, df_0_1000_35C, df_0_1000_45C,
                                  df_0_1000_55C):
    
    # Concat df
    df_0_1000_concat = pd.concat([df_0_1000_25C, df_0_1000_35C, df_0_1000_45C,
                                  df_0_1000_55C]).reset_index(drop=True)
    
    # # Color name
    # colors = ['#1b9e77','#d95f02','#7570b3']
    
    # Going through each parameter on the parameter list
    for parameter in parameter_ratio_list:
        
        sns.set(rc={'figure.figsize':(6,4)})
        plt.figure()
        
        g=sns.jointplot(data=df_0_1000_concat, x="temperature",
                        y=parameter,hue='device_name',alpha=0.5,
                        linewidth=0)
        
        plt.tight_layout()
        
        # Saving figure
        plt.savefig('figures/'+folder_run_name+'1000vs0ratio_'+parameter+'.png', dpi=600)
        
        # Zoomed in version for PCE_mpp_ratio
        if parameter=='PCE_mpp_ratio':
            
            sns.set(rc={'figure.figsize':(6,4)})
            plt.figure()
    
            g=sns.jointplot(data=df_0_1000_concat, x="temperature",
                            y=parameter,hue='device_name',alpha=0.5,
                            linewidth=0)
            
            plt.ylim([0,1.5])
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+'1000vs0ratio_'+parameter+'zoomed.png', dpi=600)


###############################################################################
# FUNCTIONS: PLOTTING (PART 1) BASIC AREA, MEAN, MAX
###############################################################################

# Plot parameters for various parameters area in different temperatures
def plot_area_parameters(df_all_25C_area, df_all_35C_area, df_all_45C_area,
                         df_all_55C_area, folder_run_name, type_params):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name):
        os.makedirs('figures/'+folder_run_name)
    
    # Select parameters type to plot
    if type_params == 'normal':
        param_list_selected = parameter_list
    elif type_params == 'loss':
        param_list_selected = parameter_loss_list
    else:
        raise ValueError('Type of parameters isnt correct!')
    
    
    # Going through each parameter on the parameter list
    for parameter in param_list_selected:
        
        # FIRST FIGURE: NOT NORMALIZED
        sns.set(rc={'figure.figsize':(7,3)})
        
        plt.figure()
    
        colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
    
        for index,row in df_all_55C_area.iterrows():
            g = sns.scatterplot(x=df_all_55C_area['area_perCycle'][index]['cycle'],
                                y=df_all_55C_area['area_perCycle'][index][parameter],
                                alpha=0.08, s=15, linewidth=0,
                                color=colors[3])
            
        for index,row in df_all_45C_area.iterrows():
            g = sns.scatterplot(x=df_all_45C_area['area_perCycle'][index]['cycle'],
                                y=df_all_45C_area['area_perCycle'][index][parameter],
                                alpha=0.15, s=15, linewidth=0,
                                color=colors[2])
            
        for index,row in df_all_35C_area.iterrows():
            g = sns.scatterplot(x=df_all_35C_area['area_perCycle'][index]['cycle'],
                                y=df_all_35C_area['area_perCycle'][index][parameter],
                                alpha=0.15, s=15, linewidth=0,
                                color=colors[1])
        
        for index,row in df_all_25C_area.iterrows():
            g = sns.scatterplot(x=df_all_25C_area['area_perCycle'][index]['cycle'],
                                y=df_all_25C_area['area_perCycle'][index][parameter],
                                alpha=0.2, s=15, linewidth=0,
                                color=colors[0])
        
        # Making the figure prettier
        plt.rcParams['font.family'] = 'Arial'
        # plt.xlabel('cycle')
        plt.ylabel(parameter)
        # plt.ylim([-0.5,17.5])
    
        plt.tight_layout()
        
        # Saving figure
        plt.savefig('figures/'+folder_run_name+'areaParam_allBatches_'+parameter+'.png', dpi=600)
        
        # # Close figure
        # plt.close('all')
        
        # SECOND FIGURE: NORMALIZED BASED ON HOUR
        sns.set(rc={'figure.figsize':(7,3)})
        
        plt.figure()
    
        colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
    
        for index,row in df_all_55C_area.iterrows():
            g = sns.scatterplot(x=df_all_55C_area['area_perCycle'][index]['cycle'],
                                y=df_all_55C_area['area_perCycle'][index][parameter]/hour_cycle_55C,
                                alpha=0.08, s=15, linewidth=0,
                                color=colors[3])
            
        for index,row in df_all_45C_area.iterrows():
            g = sns.scatterplot(x=df_all_45C_area['area_perCycle'][index]['cycle'],
                                y=df_all_45C_area['area_perCycle'][index][parameter]/hour_cycle_45C,
                                alpha=0.15, s=15, linewidth=0,
                                color=colors[2])
            
        for index,row in df_all_35C_area.iterrows():
            g = sns.scatterplot(x=df_all_35C_area['area_perCycle'][index]['cycle'],
                                y=df_all_35C_area['area_perCycle'][index][parameter]/hour_cycle_35C,
                                alpha=0.15, s=15, linewidth=0,
                                color=colors[1])
        
        for index,row in df_all_25C_area.iterrows():
            g = sns.scatterplot(x=df_all_25C_area['area_perCycle'][index]['cycle'],
                                y=df_all_25C_area['area_perCycle'][index][parameter]/hour_cycle_25C,
                                alpha=0.2, s=15, linewidth=0,
                                color=colors[0])
        
        # Making the figure prettier
        plt.rcParams['font.family'] = 'Arial'
        # plt.xlabel('cycle')
        plt.ylabel(parameter)
        # plt.ylim([-0.5,17.5])
    
        plt.tight_layout()
        
        # Saving figure
        plt.savefig('figures/'+folder_run_name+'areaParam_allBatches_'+parameter+'_perHour.png', dpi=600)
        
        # # Close figure
        # plt.close('all')
        

# Plot parameters for various parameters area in different temperatures for different batches        
def plot_area_parameters_specific_batch(df_all_25C_area, df_all_35C_area, df_all_45C_area,
                                        df_all_55C_area, folder_run_name, type_params):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name):
        os.makedirs('figures/'+folder_run_name)
        
    # Select parameters type to plot
    if type_params == 'normal':
        param_list_selected = parameter_list
    elif type_params == 'loss':
        param_list_selected = parameter_loss_list
    else:
        raise ValueError('Type of parameters isnt correct!')
    
    # Go through each batchname
    
    for batch_int in batchname_list:
        
        # Selecting a specific batch
        selected_25C = df_all_25C_area[df_all_25C_area['device_name']==batch_int]
        selected_35C = df_all_35C_area[df_all_35C_area['device_name']==batch_int]
        selected_45C = df_all_45C_area[df_all_45C_area['device_name']==batch_int]
        selected_55C = df_all_55C_area[df_all_55C_area['device_name']==batch_int]
        
        # Going through each parameter on the parameter list
        for parameter in param_list_selected:
            
            # FIRST FIGURE: NOT NORMALIZED
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
            
            for index,row in selected_55C.iterrows():
                g = sns.scatterplot(x=selected_55C['area_perCycle'][index]['cycle'],
                                    y=selected_55C['area_perCycle'][index][parameter],
                                    alpha=0.08, s=15, linewidth=0,
                                    color=colors[3])
                
            for index,row in selected_45C.iterrows():
                g = sns.scatterplot(x=selected_45C['area_perCycle'][index]['cycle'],
                                    y=selected_45C['area_perCycle'][index][parameter],
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[2])
                
            for index,row in selected_35C.iterrows():
                g = sns.scatterplot(x=selected_35C['area_perCycle'][index]['cycle'],
                                    y=selected_35C['area_perCycle'][index][parameter],
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[1])
                
            for index,row in selected_25C.iterrows():
                g = sns.scatterplot(x=selected_25C['area_perCycle'][index]['cycle'],
                                    y=selected_25C['area_perCycle'][index][parameter],
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[0])
        
            # Making the figure prettier
            plt.rcParams['font.family'] = 'Arial'
            # plt.xlabel('cycle')
            plt.ylabel(parameter)
            # plt.ylim([-0.5,17.5])
        
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+batch_int+'_areaParam_'+parameter+'.png', dpi=600)
            
            # Close figure
            plt.close('all')
            
            # SECOND FIGURE: NORMALIZED BASED ON HOUR
            # sns.set(rc={'figure.figsize':(7,3)})
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
            
            # List for regression slopes
            list_slope_55C = []
            list_slope_45C = []
            list_slope_35C = []
            list_slope_25C = []
            
            for index,row in selected_55C.iterrows():
                g = sns.scatterplot(x=selected_55C['area_perCycle'][index]['cycle'],
                                    y=selected_55C['area_perCycle'][index][parameter]/hour_cycle_55C,
                                    alpha=0.08, s=15, linewidth=0,
                                    color=colors[3])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_55C['area_perCycle'][index]['cycle'], 
                                                                         selected_55C['area_perCycle'][index][parameter]/hour_cycle_55C)
                list_slope_55C.append(slope)
                            
                
            for index,row in selected_45C.iterrows():
                g = sns.scatterplot(x=selected_45C['area_perCycle'][index]['cycle'],
                                    y=selected_45C['area_perCycle'][index][parameter]/hour_cycle_45C,
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[2])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_45C['area_perCycle'][index]['cycle'], 
                                                                         selected_45C['area_perCycle'][index][parameter]/hour_cycle_45C)
                list_slope_45C.append(slope)
                
            for index,row in selected_35C.iterrows():
                g = sns.scatterplot(x=selected_35C['area_perCycle'][index]['cycle'],
                                    y=selected_35C['area_perCycle'][index][parameter]/hour_cycle_35C,
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[1])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_35C['area_perCycle'][index]['cycle'], 
                                                                         selected_35C['area_perCycle'][index][parameter]/hour_cycle_35C)
                list_slope_35C.append(slope)
                
            for index,row in selected_25C.iterrows():
                g = sns.scatterplot(x=selected_25C['area_perCycle'][index]['cycle'],
                                    y=selected_25C['area_perCycle'][index][parameter]/hour_cycle_25C,
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[0])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_25C['area_perCycle'][index]['cycle'], 
                                                                         selected_25C['area_perCycle'][index][parameter]/hour_cycle_25C)
                list_slope_25C.append(slope)
        
            # Making the figure prettier
            plt.rcParams['font.family'] = 'Arial'
            # plt.xlabel('cycle')
            plt.ylabel(parameter+'_perHour')
            # plt.ylim([-0.5,17.5])
        
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+batch_int+'_areaParam_'+parameter+'_perHour.png', dpi=600)
            
            # Close figure
            plt.close('all')
            
            
            # THIRD FIGURE: slope regression results
            
            combined_list_slope = [list_slope_25C, list_slope_35C, list_slope_45C, list_slope_55C]
            
            # sns.set(rc={'figure.figsize':(7,3)})
            sns.set(rc={'figure.figsize':(2,3)})
            plt.figure()
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
            
            # Calculate bin edges based on the combined data
            combined_data = np.concatenate(combined_list_slope)
            bin_edges = np.histogram_bin_edges(combined_data, bins=12)
            
            # Plot histograms for each list
            for i in range(len(combined_list_slope)):
                sns.histplot(y=combined_list_slope[i], bins=bin_edges, color=colors[i], kde=True, label=f'Slope {i + 1}')
                
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+batch_int+'_areaParam_'+parameter+'_perHour_slope_histogram.png', dpi=600)
            
            # Close figure
            plt.close('all')
            

# Plot parameters for various parameters area in different temperatures for different batches        
def plot_area_parameters_specific_batch_no35C(df_all_25C_area, df_all_45C_area,
                                              df_all_55C_area, folder_run_name, type_params):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+'no35C/'):
        os.makedirs('figures/'+folder_run_name+'no35C/')
        
    # Select parameters type to plot
    if type_params == 'normal':
        param_list_selected = parameter_list
    elif type_params == 'loss':
        param_list_selected = parameter_loss_list
    else:
        raise ValueError('Type of parameters isnt correct!')
    
    # Go through each batchname
    
    for batch_int in batchname_list:
        
        # Selecting a specific batch
        selected_25C = df_all_25C_area[df_all_25C_area['device_name']==batch_int]
        # selected_35C = df_all_35C_area[df_all_35C_area['device_name']==batch_int]
        selected_45C = df_all_45C_area[df_all_45C_area['device_name']==batch_int]
        selected_55C = df_all_55C_area[df_all_55C_area['device_name']==batch_int]
        
        # Going through each parameter on the parameter list
        for parameter in param_list_selected:
            
            # FIRST FIGURE: NOT NORMALIZED
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
            
            for index,row in selected_55C.iterrows():
                g = sns.scatterplot(x=selected_55C['area_perCycle'][index]['cycle'],
                                    y=selected_55C['area_perCycle'][index][parameter],
                                    alpha=0.08, s=15, linewidth=0,
                                    color=colors[3])
                
            for index,row in selected_45C.iterrows():
                g = sns.scatterplot(x=selected_45C['area_perCycle'][index]['cycle'],
                                    y=selected_45C['area_perCycle'][index][parameter],
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[2])
                
            # for index,row in selected_35C.iterrows():
            #     g = sns.scatterplot(x=selected_35C['area_perCycle'][index]['cycle'],
            #                         y=selected_35C['area_perCycle'][index][parameter],
            #                         alpha=0.15, s=15, linewidth=0,
            #                         color=colors[1])
                
            for index,row in selected_25C.iterrows():
                g = sns.scatterplot(x=selected_25C['area_perCycle'][index]['cycle'],
                                    y=selected_25C['area_perCycle'][index][parameter],
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[0])
        
            # Making the figure prettier
            plt.rcParams['font.family'] = 'Arial'
            # plt.xlabel('cycle')
            plt.ylabel(parameter)
            # plt.ylim([-0.5,17.5])
        
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+'no35C/'+batch_int+'_areaParam_'+parameter+'.png', dpi=600)
            
            # Close figure
            plt.close('all')
            
            # SECOND FIGURE: NORMALIZED BASED ON HOUR
            # sns.set(rc={'figure.figsize':(7,3)})
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
            
            # List for regression slopes
            list_slope_55C = []
            list_slope_45C = []
            # list_slope_35C = []
            list_slope_25C = []
            
            for index,row in selected_55C.iterrows():
                g = sns.scatterplot(x=selected_55C['area_perCycle'][index]['cycle'],
                                    y=selected_55C['area_perCycle'][index][parameter]/hour_cycle_55C,
                                    alpha=0.08, s=15, linewidth=0,
                                    color=colors[3])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_55C['area_perCycle'][index]['cycle'], 
                                                                         selected_55C['area_perCycle'][index][parameter]/hour_cycle_55C)
                list_slope_55C.append(slope)
                            
                
            for index,row in selected_45C.iterrows():
                g = sns.scatterplot(x=selected_45C['area_perCycle'][index]['cycle'],
                                    y=selected_45C['area_perCycle'][index][parameter]/hour_cycle_45C,
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[2])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_45C['area_perCycle'][index]['cycle'], 
                                                                         selected_45C['area_perCycle'][index][parameter]/hour_cycle_45C)
                list_slope_45C.append(slope)
                
            # for index,row in selected_35C.iterrows():
            #     g = sns.scatterplot(x=selected_35C['area_perCycle'][index]['cycle'],
            #                         y=selected_35C['area_perCycle'][index][parameter]/hour_cycle_35C,
            #                         alpha=0.15, s=15, linewidth=0,
            #                         color=colors[1])
                
            #     # Perform linear regression
            #     slope, intercept, r_value, p_value, std_err = linregress(selected_35C['area_perCycle'][index]['cycle'], 
            #                                                              selected_35C['area_perCycle'][index][parameter]/hour_cycle_35C)
            #     list_slope_35C.append(slope)
                
            for index,row in selected_25C.iterrows():
                g = sns.scatterplot(x=selected_25C['area_perCycle'][index]['cycle'],
                                    y=selected_25C['area_perCycle'][index][parameter]/hour_cycle_25C,
                                    alpha=0.15, s=15, linewidth=0,
                                    color=colors[0])
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(selected_25C['area_perCycle'][index]['cycle'], 
                                                                         selected_25C['area_perCycle'][index][parameter]/hour_cycle_25C)
                list_slope_25C.append(slope)
        
            # Making the figure prettier
            plt.rcParams['font.family'] = 'Arial'
            # plt.xlabel('cycle')
            plt.ylabel(parameter+'_perHour')
            # plt.ylim([-0.5,17.5])
        
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+'no35C/'+batch_int+'_areaParam_'+parameter+'_perHour.png', dpi=600)
            
            # Close figure
            plt.close('all')
            
            
            # THIRD FIGURE: slope regression results
            
            combined_list_slope = [list_slope_25C, list_slope_45C, list_slope_55C] #list_slope_35C
            
            # sns.set(rc={'figure.figsize':(7,3)})
            sns.set(rc={'figure.figsize':(2,3)})
            plt.figure()
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
            
            # Calculate bin edges based on the combined data
            combined_data = np.concatenate(combined_list_slope)
            bin_edges = np.histogram_bin_edges(combined_data, bins=12)
            
            # Plot histograms for each list
            for i in range(len(combined_list_slope)):
                sns.histplot(y=combined_list_slope[i], bins=bin_edges, color=colors[i], kde=True, label=f'Slope {i + 1}')
                
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+'no35C/'+batch_int+'_areaParam_'+parameter+'_perHour_slope_histogram.png', dpi=600)
            
            # Close figure
            plt.close('all')

            
def plot_area_parameters_stats_specific_batch(df_all_25C_area, df_all_35C_area, df_all_45C_area,
                                              df_all_55C_area, folder_run_name):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name):
        os.makedirs('figures/'+folder_run_name)
        
    
    # Go through each batchname     
    for batch_int in batchname_list:
        
        # Selecting a specific batch
        selected_25C = df_stats_25C[df_stats_25C['device_name']==batch_int]
        selected_35C = df_stats_35C[df_stats_35C['device_name']==batch_int]
        selected_45C = df_stats_45C[df_stats_45C['device_name']==batch_int]
        selected_55C = df_stats_55C[df_stats_55C['device_name']==batch_int]
        
        # Going through each parameter on the parameter list
        count = 0
        
        for parameter in parameter_mean_list:
        #     sns.set(rc={'figure.figsize':(7,3)})
            plt.figure()
            fig, ax = plt.subplots(figsize=(7,3))
            colors=['#fdcc8a','#fc8d59','#e34a33','#b30000']
        
            for index,row in selected_25C.iterrows():
                ax.plot(selected_25C['area_perCycle_stats'][index]['cycle'],
                        selected_25C['area_perCycle_stats'][index][parameter],
                        color = colors[0],
                        label = '25C')
                
                # Calculate lower and upper
                lower = selected_25C['area_perCycle_stats'][index][parameter]-selected_25C['area_perCycle_stats'][index][parameter_std_list[count]]
                upper = selected_25C['area_perCycle_stats'][index][parameter]+selected_25C['area_perCycle_stats'][index][parameter_std_list[count]]
                
                ax.plot(selected_25C['area_perCycle_stats'][index]['cycle'],lower,color=colors[0],alpha=0.1)
                ax.plot(selected_25C['area_perCycle_stats'][index]['cycle'],upper,color=colors[0],alpha=0.1)
                ax.fill_between(selected_25C['area_perCycle_stats'][index]['cycle'],lower,upper,color=colors[0],alpha=0.4)
            
            for index,row in selected_35C.iterrows():
                ax.plot(selected_35C['area_perCycle_stats'][index]['cycle'],
                        selected_35C['area_perCycle_stats'][index][parameter],
                        color = colors[1],
                        label = '35C')
                
                # Calculate lower and upper
                lower = selected_35C['area_perCycle_stats'][index][parameter]-selected_35C['area_perCycle_stats'][index][parameter_std_list[count]]
                upper = selected_35C['area_perCycle_stats'][index][parameter]+selected_35C['area_perCycle_stats'][index][parameter_std_list[count]]
                
                ax.plot(selected_35C['area_perCycle_stats'][index]['cycle'],lower,color=colors[1],alpha=0.1)
                ax.plot(selected_35C['area_perCycle_stats'][index]['cycle'],upper,color=colors[1],alpha=0.1)
                ax.fill_between(selected_35C['area_perCycle_stats'][index]['cycle'],lower,upper,color=colors[1],alpha=0.4)
            
            for index,row in selected_45C.iterrows():
                ax.plot(selected_45C['area_perCycle_stats'][index]['cycle'],
                        selected_45C['area_perCycle_stats'][index][parameter],
                        color = colors[2],
                        label = '45C')
                
                # Calculate lower and upper
                lower = selected_45C['area_perCycle_stats'][index][parameter]-selected_45C['area_perCycle_stats'][index][parameter_std_list[count]]
                upper = selected_45C['area_perCycle_stats'][index][parameter]+selected_45C['area_perCycle_stats'][index][parameter_std_list[count]]
                
                ax.plot(selected_45C['area_perCycle_stats'][index]['cycle'],lower,color=colors[2],alpha=0.1)
                ax.plot(selected_45C['area_perCycle_stats'][index]['cycle'],upper,color=colors[2],alpha=0.1)
                ax.fill_between(selected_45C['area_perCycle_stats'][index]['cycle'],lower,upper,color=colors[2],alpha=0.4)
        
            for index,row in selected_55C.iterrows():
                ax.plot(selected_55C['area_perCycle_stats'][index]['cycle'],
                        selected_55C['area_perCycle_stats'][index][parameter],
                        color = colors[3],
                        label = '55C')
                
                # Calculate lower and upper
                lower = selected_55C['area_perCycle_stats'][index][parameter]-selected_55C['area_perCycle_stats'][index][parameter_std_list[count]]
                upper = selected_55C['area_perCycle_stats'][index][parameter]+selected_55C['area_perCycle_stats'][index][parameter_std_list[count]]
                
                ax.plot(selected_55C['area_perCycle_stats'][index]['cycle'],lower,color=colors[3],alpha=0.1)
                ax.plot(selected_55C['area_perCycle_stats'][index]['cycle'],upper,color=colors[3],alpha=0.1)
                ax.fill_between(selected_55C['area_perCycle_stats'][index]['cycle'],lower,upper,color=colors[3],alpha=0.4)
            
            count+=1
            # Making the figure prettier
            plt.rcParams['font.family'] = 'Arial'
            plt.xlabel('cycle')
            plt.ylabel(parameter)
            # plt.ylim([-0.5,17.5])
        
            plt.tight_layout()
            
            # Saving figure
            plt.savefig('figures/'+folder_run_name+batch_int+'_areaParam_mean_std_'+parameter+'.png', dpi=600)

            # # Close figure
            # plt.close('all')
                

###############################################################################
# FUNCTIONS: REGRESSION FOR PARAMETERS (sns.regplot) + ARRHENIUS
###############################################################################..

# Go through batch and temperature, and do regression for each of them
def calculate_regression_specificT_specificBatch(df_all_25C_area, df_all_35C_area,
                                                 df_all_45C_area, df_all_55C_area,
                                                 type_reg): # type_reg = 'specific_batch', and 'all_batches'
    
    # Selecting a specific batch based on the type of regression being done
    
    if type_reg == 'specific_batch':
    
        # Go through a specific batch
        for batch_int in batchname_list:
            
            selected_25C = df_all_25C_area[df_all_25C_area['device_name']==batch_int]
            selected_35C = df_all_35C_area[df_all_35C_area['device_name']==batch_int]
            selected_45C = df_all_45C_area[df_all_45C_area['device_name']==batch_int]
            selected_55C = df_all_55C_area[df_all_55C_area['device_name']==batch_int]
            
            list_selected = [selected_25C, selected_35C, selected_45C, selected_55C]
            
            dict_temp = {0:'25C',
                         1:'35C',
                         2:'45C',
                         3:'55C'}
            
            counter = 0 # to count for dict_temp
            
            # Go through the temperatures
            for selected_temp in list_selected:
                
                temp_int = dict_temp[counter]
                counter += 1
                
                # Remake the dataframe
                dummy_df = pd.DataFrame()
                
                for index,row in selected_temp.iterrows():
                    
                    # Make a concat df
                    concat_df = selected_temp['area_perCycle'][index]
                    concat_df['device_name'] = selected_temp['device_name'][index]
                    concat_df['pixel'] = selected_temp['pixel'][index]
                    concat_df['temperature'] = selected_temp['temperature'][index]
                    concat_df['hour_cycle'] = selected_temp['hour_cycle'][index]
                    
                    # Concatenate
                    dummy_df = pd.concat([dummy_df, concat_df], ignore_index=True)
                
                # Calculate normalized, per area parameters
                dummy_df['PCEmppArea_perCycle_perHour']=dummy_df['PCEmppArea_perCycle']/dummy_df['hour_cycle']
                dummy_df['VmppArea_perCycle_perHour']=dummy_df['VmppArea_perCycle']/dummy_df['hour_cycle']
                dummy_df['PCEmppAreaLoss_perCycle_perHour']=dummy_df['PCEmppAreaLoss_perCycle']/dummy_df['hour_cycle']
                dummy_df['VmppAreaLoss_perCycle_perHour']=dummy_df['VmppAreaLoss_perCycle']/dummy_df['hour_cycle']
            
                # Empty list to store results from regressions
                batch_int_list = []
                parameter_int_list = []
                slope1_list = []
                intercept1_list = []
                r1_list = []
                p1_list = []
                sterr1_list = []
                
                # Plot regression for first 4 parameters
                for i in range(4): # There are 4 parameters
            
                    # New figure
                    sns.set(rc={'figure.figsize':(5,3)})
                    
                    plt.figure()
                    
                    # Plot the overview and OLS trendline
                    ax = sns.regplot(data=dummy_df, x='cycle',
                                    y=(parameter_list[i]+'_perHour'),
                                    fit_reg=True,ci = 99,
                                    scatter_kws={'alpha':0.2}, # "color": "blue",
                                    line_kws={"color": "red"})#,'alpha':0.8})
                
                    plt.setp(ax.collections[1], alpha=0.3)
                
                    #calculate slope and intercept of regression equation
                    slope1, intercept1, r1, p1, sterr1 = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                                                y=ax.get_lines()[0].get_ydata())
                    # Append to the lists
                    batch_int_list.append(batch_int)
                    parameter_int_list.append(parameter_list[i])
                    slope1_list.append(slope1)
                    intercept1_list.append(intercept1)
                    r1_list.append(r1)
                    p1_list.append(p1)
                    sterr1_list.append(sterr1)
                    
                    
                    # print('For ',batch_int, ' ',parameter_list[i],'_perHour; slope: ',
                    #       slope1,', intercept: ', intercept1)#,', p1: ', r1)
                
                    # Saving figure
                    plt.savefig('figures/'+folder_run_name+batch_int+'_'+temp_int+'_'+
                                parameter_list[i]+'_perHour_snsreg.png', dpi=600)
                    
                    # plt.savefig('figures/'+folder_run_name+'areaParam_allBatches_'+parameter+'_perHour.png', dpi=600)
                    
                
                # Plot regression for first 4 parameters, go to a different set of parameters (5-8)
                for i in np.arange(4,8,1):
                    
                    # New figure
                    sns.set(rc={'figure.figsize':(5,3)})
                    
                    plt.figure()
                    
                    # Plot the overview and OLS trendline
                    ax = sns.regplot(data=dummy_df, x='cycle',
                                    y=(parameter_list[i]),
                                    fit_reg=True,ci = 99,
                                    scatter_kws={'alpha':0.2}, # "color": "blue",
                                    line_kws={"color": "red"})#,'alpha':0.8})
                
                    plt.setp(ax.collections[1], alpha=0.3)
                
                    #calculate slope and intercept of regression equation
                    slope1, intercept1, r1, p1, sterr1 = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                                                y=ax.get_lines()[0].get_ydata())
                    # Append to the lists
                    batch_int_list.append(batch_int)
                    parameter_int_list.append(parameter_list[i])
                    slope1_list.append(slope1)
                    intercept1_list.append(intercept1)
                    r1_list.append(r1)
                    p1_list.append(p1)
                    sterr1_list.append(sterr1)
                    
                    # print('For ',batch_int, ' ',parameter_list[i],'; slope: ',
                    #       slope1,', intercept: ', intercept1)#,', p1: ', r1)
                
                    # Saving figure
                    plt.savefig('figures/'+folder_run_name+batch_int+'_'+temp_int+'_'+
                                parameter_list[i]+'_perHour_snsreg.png', dpi=600)
                    
                    # plt.savefig('figures/20230911_'+batch_int+'_35C_'+parameter_list[i]+'_snsreg.png', dpi=600)
                
                # Save list as df
                regression_data = {'device_name':batch_int_list,
                                   'parameter':parameter_int_list,
                                   'reg_slope':slope1_list,
                                   'reg_intercept':intercept1_list,
                                   'reg_r':r1_list,
                                   'reg_p':p1_list,
                                   'reg_sterr:':sterr1_list}
                
                regression_data_df = pd.DataFrame(regression_data)
                
                # Save df as csv
                regression_data_df.to_csv('figures/'+folder_run_name+batch_int+'_'+
                                          temp_int+'_regression_results.csv',
                                          index=False)
                
                # Save dummy_df as csv
                dummy_df.to_csv('figures/'+folder_run_name+batch_int+'_'+
                                temp_int+'_regression_normalized_perArea_parameters.csv',
                                index=False)
        
    elif type_reg == 'all_batches':
        
        selected_25C = df_all_25C_area
        selected_35C = df_all_35C_area
        selected_45C = df_all_45C_area
        selected_55C = df_all_55C_area
        
        list_selected = [selected_25C, selected_35C, selected_45C, selected_55C]
        
        dict_temp = {0:'25C',
                     1:'35C',
                     2:'45C',
                     3:'55C'}
        
        counter = 0 # to count for dict_temp
        
        # Go through the temperatures
        for selected_temp in list_selected:
            
            temp_int = dict_temp[counter]
            counter += 1
            
            # Remake the dataframe
            dummy_df = pd.DataFrame()
            
            for index,row in selected_temp.iterrows():
                
                # Make a concat df
                concat_df = selected_temp['area_perCycle'][index]
                concat_df['device_name'] = selected_temp['device_name'][index]
                concat_df['pixel'] = selected_temp['pixel'][index]
                concat_df['temperature'] = selected_temp['temperature'][index]
                concat_df['hour_cycle'] = selected_temp['hour_cycle'][index]
                
                # Concatenate
                dummy_df = pd.concat([dummy_df, concat_df], ignore_index=True)
            
            # Calculate normalized, per area parameters
            dummy_df['PCEmppArea_perCycle_perHour']=dummy_df['PCEmppArea_perCycle']/dummy_df['hour_cycle']
            dummy_df['VmppArea_perCycle_perHour']=dummy_df['VmppArea_perCycle']/dummy_df['hour_cycle']
            dummy_df['PCEmppAreaLoss_perCycle_perHour']=dummy_df['PCEmppAreaLoss_perCycle']/dummy_df['hour_cycle']
            dummy_df['VmppAreaLoss_perCycle_perHour']=dummy_df['VmppAreaLoss_perCycle']/dummy_df['hour_cycle']
        
            # Empty list to store results from regressions
            parameter_int_list = []
            slope1_list = []
            intercept1_list = []
            r1_list = []
            p1_list = []
            sterr1_list = []
            
            # Plot regression for first 4 parameters
            for i in range(4): # There are 4 parameters
        
                # New figure
                sns.set(rc={'figure.figsize':(5,3)})
                
                plt.figure()
                
                # Plot the overview and OLS trendline
                ax = sns.regplot(data=dummy_df, x='cycle',
                                y=(parameter_list[i]+'_perHour'),
                                fit_reg=True,ci = 99,
                                scatter_kws={'alpha':0.2}, # "color": "blue",
                                line_kws={"color": "red"})#,'alpha':0.8})
            
                plt.setp(ax.collections[1], alpha=0.3)
            
                #calculate slope and intercept of regression equation
                slope1, intercept1, r1, p1, sterr1 = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                                            y=ax.get_lines()[0].get_ydata())
                # Append to the lists
                parameter_int_list.append(parameter_list[i])
                slope1_list.append(slope1)
                intercept1_list.append(intercept1)
                r1_list.append(r1)
                p1_list.append(p1)
                sterr1_list.append(sterr1)
                
                
                # print('For ',batch_int, ' ',parameter_list[i],'_perHour; slope: ',
                #       slope1,', intercept: ', intercept1)#,', p1: ', r1)
            
                # Saving figure
                plt.savefig('figures/'+folder_run_name+'_allbatches_'+temp_int+'_'+
                            parameter_list[i]+'_perHour_snsreg.png', dpi=600)
                
                # plt.savefig('figures/'+folder_run_name+'areaParam_allBatches_'+parameter+'_perHour.png', dpi=600)
                
            
            # Plot regression for first 4 parameters, go to a different set of parameters (5-8)
            for i in np.arange(4,8,1):
                
                # New figure
                sns.set(rc={'figure.figsize':(5,3)})
                
                plt.figure()
                
                # Plot the overview and OLS trendline
                ax = sns.regplot(data=dummy_df, x='cycle',
                                y=(parameter_list[i]),
                                fit_reg=True,ci = 99,
                                scatter_kws={'alpha':0.2}, # "color": "blue",
                                line_kws={"color": "red"})#,'alpha':0.8})
            
                plt.setp(ax.collections[1], alpha=0.3)
            
                #calculate slope and intercept of regression equation
                slope1, intercept1, r1, p1, sterr1 = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                                            y=ax.get_lines()[0].get_ydata())
                # Append to the lists
                parameter_int_list.append(parameter_list[i])
                slope1_list.append(slope1)
                intercept1_list.append(intercept1)
                r1_list.append(r1)
                p1_list.append(p1)
                sterr1_list.append(sterr1)
                
                # print('For ',batch_int, ' ',parameter_list[i],'; slope: ',
                #       slope1,', intercept: ', intercept1)#,', p1: ', r1)
            
                # Saving figure
                plt.savefig('figures/'+folder_run_name+'_allbatches_'+temp_int+'_'+
                            parameter_list[i]+'_perHour_snsreg.png', dpi=600)
                
                # plt.savefig('figures/20230911_'+batch_int+'_35C_'+parameter_list[i]+'_snsreg.png', dpi=600)
            
            # Save list as df
            regression_data = {'parameter':parameter_int_list,
                               'reg_slope':slope1_list,
                               'reg_intercept':intercept1_list,
                               'reg_r':r1_list,
                               'reg_p':p1_list,
                               'reg_sterr:':sterr1_list}
            
            regression_data_df = pd.DataFrame(regression_data)
            
            # Save df as csv
            regression_data_df.to_csv('figures/'+folder_run_name+'_allbatches_'+
                                      temp_int+'_regression_results.csv',
                                      index=False)
            
            # Save dummy_df as csv
            dummy_df.to_csv('figures/'+folder_run_name+'_allbatches_'+
                            temp_int+'_regression_normalized_perArea_parameters.csv',
                            index=False)
        
    else:
        raise ValueError('Type of regressions isnt correct!')  

# Function to plot and fit, return the fitting
def plot_fit_regression(df,x,y,device,pixel,temp):
    # sns.set(rc={'figure.figsize':(5,3)})
    plt.figure()

    # Plot the overview and OLS trendline
    ax = sns.regplot(data=df, x=x,
                     y=y,
                     fit_reg=True,ci = 95,
                     scatter_kws={'alpha':0.4}, # "color": "blue",
                     line_kws={"color": "red"})#,'alpha':0.8})

    # Calculate slope and intercept of regression equation
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                           y=ax.get_lines()[0].get_ydata())
    
    plt.tight_layout()
    
    # Save figure
#     plt.savefig('figures/20230531_fitting/20230531_'+device+'_'+pixel+'_'+str(temp)+'_'+x+'_'+y+'.png', dpi=600)
    
    plt.close()
    
    return slope, intercept, r, p, sterr

def fit_params_regression(df, x, param1, param2, list1slope, list1intercept, list2slope, list2intercept,device,pixel,temp):
       
    # Plot and fit
    slope, intercept, r, p, sterr = plot_fit_regression(df,x,param1,device,pixel,temp)

    # Append to the list
    list1slope.append(slope)
    list1intercept.append(intercept)

    # V mpp #            
    # Plot and fit
    slope, intercept, r, p, sterr = plot_fit_regression(df,x,param2,device,pixel,temp)

    # Append to the list
    list2slope.append(slope)
    list2intercept.append(intercept)    

# Calculate sns.regplot for each pixel
def calculate_regression_eachpixel(df_all_25C_35C_45C_55C_area):
    
    # Remake the data frame (append, long), unpacking
    dummy_df = pd.DataFrame()
    dummy_df
    
    for index,row in df_all_25C_35C_45C_55C_area.iterrows():
    #     dummy_df=dummy_df.append(selected_45C['area_perCycle'][index])
    #     device_name = selected_45C['device_name'][index]
        
        # Make a concat df
        concat_df = df_all_25C_35C_45C_55C_area['area_perCycle'][index]
        concat_df['device_name'] = df_all_25C_35C_45C_55C_area['device_name'][index]
        concat_df['pixel'] = df_all_25C_35C_45C_55C_area['pixel'][index]
        concat_df['temperature'] = df_all_25C_35C_45C_55C_area['temperature'][index]
        concat_df['hour_cycle'] = df_all_25C_35C_45C_55C_area['hour_cycle'][index]
        
        # Concatenate
        dummy_df = pd.concat([dummy_df, concat_df], ignore_index=True)
        
    # Calculate normalized, per area parameters
    dummy_df['PCEmppArea_perCycle_perHour']=dummy_df['PCEmppArea_perCycle']/dummy_df['hour_cycle']
    dummy_df['VmppArea_perCycle_perHour']=dummy_df['VmppArea_perCycle']/dummy_df['hour_cycle']
    dummy_df['PCEmppAreaLoss_perCycle_perHour']=dummy_df['PCEmppAreaLoss_perCycle']/dummy_df['hour_cycle']
    dummy_df['VmppAreaLoss_perCycle_perHour']=dummy_df['VmppAreaLoss_perCycle']/dummy_df['hour_cycle']

    
    # Introduce empty lists for the loop
    cycle_PCEmppArea_norm_slope = []
    cycle_PCEmppArea_norm_intercept = []
    cycle_VmppArea_norm_slope = []
    cycle_VmppArea_norm_intercept = []
    time_PCEmppArea_norm_slope = []
    time_PCEmppArea_norm_intercept = []
    time_VmppArea_norm_slope = []
    time_VmppArea_norm_intercept = []
    
    # Recently added loss normalized
    time_PCEmppAreaLoss_norm_slope = []
    time_PCEmppAreaLoss_norm_intercept = []
    time_VmppAreaLoss_norm_slope = []
    time_VmppAreaLoss_norm_intercept = []
    cycle_PCEmppAreaLoss_norm_slope = []
    cycle_PCEmppAreaLoss_norm_intercept = []
    cycle_VmppAreaLoss_norm_slope = []
    cycle_VmppAreaLoss_norm_intercept = []
    
    cycle_PCEmppArea_slope = []
    cycle_PCEmppArea_intercept = []
    cycle_VmppArea_slope = []
    cycle_VmppArea_intercept = []
    time_PCEmppArea_slope = []
    time_PCEmppArea_intercept = []
    time_VmppArea_slope = []
    time_VmppArea_intercept = []
    
    # Recently added loss non normalized
    cycle_PCEmppAreaLoss_slope = []
    cycle_PCEmppAreaLoss_intercept = []
    cycle_VmppAreaLoss_slope = []
    cycle_VmppAreaLoss_intercept = []
    time_PCEmppAreaLoss_slope = []
    time_PCEmppAreaLoss_intercept = []
    time_VmppAreaLoss_slope = []
    time_VmppAreaLoss_intercept = []
    
    ################
    # Recently added maxPCE non-normalized
    cycle_maxPCEmpp_slope = []
    cycle_maxPCEmpp_intercept = []
    cycle_maxVmpp_slope = []
    cycle_maxVmpp_intercept = []
    time_maxPCEmpp_slope = []
    time_maxPCEmpp_intercept = []
    time_maxVmpp_slope = []
    time_maxVmpp_intercept = []
    
    # Recently added maxPCE normalized
    cycle_maxPCEmpp_norm_slope = []
    cycle_maxPCEmpp_norm_intercept = []
    cycle_maxVmpp_norm_slope = []
    cycle_maxVmpp_norm_intercept = []
    time_maxPCEmpp_norm_slope = []
    time_maxPCEmpp_norm_intercept = []
    time_maxVmpp_norm_slope = []
    time_maxVmpp_norm_intercept = []
    
    device_name_list = []
    pixel_list = []
    temperature_list = []
    hour_cycle_list = []
    
    # Create df for fitting
    
    df_fitting = df_all_25C_35C_45C_55C_area[['device_name', 'pixel', 'temperature', 'hour_cycle']]
    
    # Go through each device name
    device_name_unique = dummy_df['device_name'].unique()
    
    for device in device_name_unique:
        selected_df = dummy_df[dummy_df['device_name']==device]
        
        # Go through each pixel
        pixel_unique = selected_df['pixel'].unique()
        
        for pixel in pixel_unique:
            
            selected_pixel_df = selected_df[selected_df['pixel']==pixel]
            
            # Go through each temperature
            temperature_unique = selected_pixel_df['temperature'].unique()
            
            for temp in temperature_unique:
                
                selected_temp_pixel_df = selected_pixel_df[selected_pixel_df['temperature']==temp]
            
                # Append information
                device_name_list.append(device)
                pixel_list.append(pixel)
                temperature_list.append(temp)
                hour_cycle_list.append(selected_temp_pixel_df['hour_cycle'].iloc[0])
    
                # Calculate normalized parameters based on the top 5
                PCEmppArea_perCycle_perHour_mean_nlargest = (selected_temp_pixel_df.nlargest(5,'PCEmppArea_perCycle_perHour'))['PCEmppArea_perCycle_perHour'].mean() # top 5 
                VmppArea_perCycle_perHour_mean_nlargest = (selected_temp_pixel_df.nlargest(5,'VmppArea_perCycle_perHour'))['VmppArea_perCycle_perHour'].mean() # top 5
                
                selected_temp_pixel_df['PCEmppArea_perCycle_perHour_norm']=selected_temp_pixel_df['PCEmppArea_perCycle_perHour']/PCEmppArea_perCycle_perHour_mean_nlargest
                selected_temp_pixel_df['VmppArea_perCycle_perHour_norm']=selected_temp_pixel_df['VmppArea_perCycle_perHour']/VmppArea_perCycle_perHour_mean_nlargest
                
                PCEmppAreaLoss_perCycle_perHour_mean_nlargest = (selected_temp_pixel_df.nlargest(5,'PCEmppAreaLoss_perCycle_perHour'))['PCEmppAreaLoss_perCycle_perHour'].mean() # top 5 
                VmppAreaLoss_perCycle_perHour_mean_nlargest = (selected_temp_pixel_df.nlargest(5,'VmppAreaLoss_perCycle_perHour'))['VmppAreaLoss_perCycle_perHour'].mean() # top 5
                
                selected_temp_pixel_df['PCEmppAreaLoss_perCycle_perHour_norm']=selected_temp_pixel_df['PCEmppAreaLoss_perCycle_perHour']/PCEmppAreaLoss_perCycle_perHour_mean_nlargest
                selected_temp_pixel_df['VmppAreaLoss_perCycle_perHour_norm']=selected_temp_pixel_df['VmppAreaLoss_perCycle_perHour']/VmppAreaLoss_perCycle_perHour_mean_nlargest
                
                max_PCEmpp_mean_nlargest = (selected_temp_pixel_df.nlargest(5,'max_PCEmpp'))['max_PCEmpp'].mean() # top 5 
                max_Vmpp_mean_nlargest = (selected_temp_pixel_df.nlargest(5,'max_Vmpp'))['max_Vmpp'].mean() # top 5
                
                selected_temp_pixel_df['max_PCEmpp_norm']=selected_temp_pixel_df['max_PCEmpp']/max_PCEmpp_mean_nlargest
                selected_temp_pixel_df['max_Vmpp_norm']=selected_temp_pixel_df['max_Vmpp']/max_Vmpp_mean_nlargest
    
                ##### FIGURE & FITTING #####
    
                ##### x-axis = cycle #####
            
                ### PCEmppArea
                # Normalized
                fit_params_regression(selected_temp_pixel_df, 'cycle', 'PCEmppArea_perCycle_perHour_norm',
                                      'VmppArea_perCycle_perHour_norm', 
                                      cycle_PCEmppArea_norm_slope,cycle_PCEmppArea_norm_intercept,
                                      cycle_VmppArea_norm_slope, cycle_VmppArea_norm_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle', 'PCEmppAreaLoss_perCycle_perHour_norm',
                                      'VmppAreaLoss_perCycle_perHour_norm', 
                                      cycle_PCEmppAreaLoss_norm_slope,cycle_PCEmppAreaLoss_norm_intercept,
                                      cycle_VmppAreaLoss_norm_slope, cycle_VmppAreaLoss_norm_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle', 'max_PCEmpp_norm',
                                      'max_Vmpp_norm', 
                                      cycle_maxPCEmpp_norm_slope,cycle_maxPCEmpp_norm_intercept,
                                      cycle_maxVmpp_norm_slope, cycle_maxVmpp_norm_intercept,
                                      device,pixel,temp)
                
                # Not normalized
                fit_params_regression(selected_temp_pixel_df, 'cycle', 'PCEmppArea_perCycle_perHour',
                                      'VmppArea_perCycle_perHour', 
                                      cycle_PCEmppArea_slope, cycle_PCEmppArea_intercept,
                                      cycle_VmppArea_slope, cycle_VmppArea_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle', 'PCEmppAreaLoss_perCycle_perHour',
                                      'VmppAreaLoss_perCycle_perHour', 
                                      cycle_PCEmppAreaLoss_slope, cycle_PCEmppAreaLoss_intercept,
                                      cycle_VmppAreaLoss_slope, cycle_VmppAreaLoss_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle', 'max_PCEmpp',
                                      'max_Vmpp', 
                                      cycle_maxPCEmpp_slope, cycle_maxPCEmpp_intercept,  
                                      cycle_maxVmpp_slope, cycle_maxVmpp_intercept,
                                      device,pixel,temp)
                
                ##### x-axis = TIME #####
                
                ### PCEmppArea
                # Normalized
                fit_params_regression(selected_temp_pixel_df, 'cycle_h', 'PCEmppArea_perCycle_perHour_norm',
                                      'VmppArea_perCycle_perHour_norm', 
                                      time_PCEmppArea_norm_slope,time_PCEmppArea_norm_intercept,
                                      time_VmppArea_norm_slope, time_VmppArea_norm_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle_h', 'PCEmppAreaLoss_perCycle_perHour_norm',
                                      'VmppAreaLoss_perCycle_perHour_norm', 
                                      time_PCEmppAreaLoss_norm_slope,time_PCEmppAreaLoss_norm_intercept,
                                      time_VmppAreaLoss_norm_slope, time_VmppAreaLoss_norm_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle_h', 'max_PCEmpp_norm',
                                      'max_Vmpp_norm', 
                                      time_maxPCEmpp_norm_slope,time_maxPCEmpp_norm_intercept,
                                      time_maxVmpp_norm_slope, time_maxVmpp_norm_intercept,
                                      device,pixel,temp)
                
                # Not normalized
                fit_params_regression(selected_temp_pixel_df, 'cycle_h', 'PCEmppArea_perCycle_perHour',
                                      'VmppArea_perCycle_perHour', 
                                      time_PCEmppArea_slope, time_PCEmppArea_intercept,
                                      time_VmppArea_slope, time_VmppArea_intercept,
                                      device,pixel,temp)
    
                fit_params_regression(selected_temp_pixel_df, 'cycle_h', 'PCEmppAreaLoss_perCycle_perHour',
                                      'VmppAreaLoss_perCycle_perHour', 
                                      time_PCEmppAreaLoss_slope, time_PCEmppAreaLoss_intercept,
                                      time_VmppAreaLoss_slope, time_VmppAreaLoss_intercept,
                                      device,pixel,temp)
                
                fit_params_regression(selected_temp_pixel_df, 'cycle_h', 'max_PCEmpp',
                                      'max_Vmpp', 
                                      time_maxPCEmpp_slope, time_maxPCEmpp_intercept,
                                      time_maxVmpp_slope, time_maxVmpp_intercept,
                                      device,pixel,temp)
                
                
    # Convert list to df
    df_fitting = pd.DataFrame(list(zip(device_name_list,pixel_list,temperature_list,hour_cycle_list,
                                       cycle_PCEmppArea_norm_slope,cycle_PCEmppArea_norm_intercept,
                                       cycle_VmppArea_norm_slope,cycle_VmppArea_norm_intercept,
                                       time_PCEmppArea_norm_slope,time_PCEmppArea_norm_intercept,
                                       time_VmppArea_norm_slope,time_VmppArea_norm_intercept,
                                       cycle_PCEmppArea_slope,cycle_PCEmppArea_intercept,
                                       cycle_VmppArea_slope,cycle_VmppArea_intercept,
                                       time_PCEmppArea_slope,time_PCEmppArea_intercept,
                                       time_VmppArea_slope,time_VmppArea_intercept,
                                       # Starting from here: Loss
                                       cycle_PCEmppAreaLoss_norm_slope,cycle_PCEmppAreaLoss_norm_intercept,
                                       cycle_VmppAreaLoss_norm_slope,cycle_VmppAreaLoss_norm_intercept,
                                       time_PCEmppAreaLoss_norm_slope,time_PCEmppAreaLoss_norm_intercept,
                                       time_VmppAreaLoss_norm_slope,time_VmppAreaLoss_norm_intercept,
                                       cycle_PCEmppAreaLoss_slope,cycle_PCEmppAreaLoss_intercept,
                                       cycle_VmppAreaLoss_slope,cycle_VmppAreaLoss_intercept,
                                       time_PCEmppAreaLoss_slope,time_PCEmppAreaLoss_intercept,
                                       time_VmppAreaLoss_slope,time_VmppAreaLoss_intercept,
                                       # Starting from here: max PCEmpp and max Vmpp
                                       cycle_maxPCEmpp_norm_slope,cycle_maxPCEmpp_norm_intercept,
                                       cycle_maxVmpp_norm_slope,cycle_maxVmpp_norm_intercept,
                                       time_maxPCEmpp_norm_slope,time_maxPCEmpp_norm_intercept,
                                       time_maxVmpp_norm_slope,time_maxVmpp_norm_intercept,
                                       cycle_maxPCEmpp_slope,cycle_maxPCEmpp_intercept,
                                       cycle_maxVmpp_slope,cycle_maxVmpp_intercept,
                                       time_maxPCEmpp_slope,time_maxPCEmpp_intercept,
                                       time_maxVmpp_slope,time_maxVmpp_intercept,
                                      )),
                              columns=['device_name','pixel','temperature','hour_cycle',
                                       'cycle_PCEmppArea_norm_slope','cycle_PCEmppArea_norm_intercept',
                                       'cycle_VmppArea_norm_slope','cycle_VmppArea_norm_intercept',
                                       'time_PCEmppArea_norm_slope','time_PCEmppArea_norm_intercept',
                                       'time_VmppArea_norm_slope','time_VmppArea_norm_intercept',
                                       'cycle_PCEmppArea_slope','cycle_PCEmppArea_intercept',
                                       'cycle_VmppArea_slope','cycle_VmppArea_intercept',
                                       'time_PCEmppArea_slope','time_PCEmppArea_intercept',
                                       'time_VmppArea_slope','time_VmppArea_intercept',
                                       # Starting from here: Loss
                                       'cycle_PCEmppAreaLoss_norm_slope','cycle_PCEmppAreaLoss_norm_intercept',
                                       'cycle_VmppAreaLoss_norm_slope','cycle_VmppAreaLoss_norm_intercept',
                                       'time_PCEmppAreaLoss_norm_slope','time_PCEmppAreaLoss_norm_intercept',
                                       'time_VmppAreaLoss_norm_slope','time_VmppAreaLoss_norm_intercept',
                                       'cycle_PCEmppAreaLoss_slope','cycle_PCEmppAreaLoss_intercept',
                                       'cycle_VmppAreaLoss_slope','cycle_VmppAreaLoss_intercept',
                                       'time_PCEmppAreaLoss_slope','time_PCEmppAreaLoss_intercept',
                                       'time_VmppAreaLoss_slope','time_VmppAreaLoss_intercept',
                                       # Starting from here: max PCEmpp and max Vmpp
                                       'cycle_maxPCEmpp_norm_slope','cycle_maxPCEmpp_norm_intercept',
                                       'cycle_maxVmpp_norm_slope','cycle_maxVmpp_norm_intercept',
                                       'time_maxPCEmpp_norm_slope','time_maxPCEmpp_norm_intercept',
                                       'time_maxVmpp_norm_slope','time_maxVmpp_norm_intercept',
                                       'cycle_maxPCEmpp_slope','cycle_maxPCEmpp_intercept',
                                       'cycle_maxVmpp_slope','cycle_maxVmpp_intercept',
                                       'time_maxPCEmpp_slope','time_maxPCEmpp_intercept',
                                       'time_maxVmpp_slope','time_maxVmpp_intercept',
                                      ])              
    
    # Create folder if not exists yet
    if not os.path.exists('output_dataframe/'+folder_run_name):
        os.makedirs('output_dataframe/'+folder_run_name)
    
    # Save df_fitting to csv
    df_fitting.to_csv('output_dataframe/'+folder_run_name+
                      'df_fitting_slope_intercept.csv',index=False)
    
    return df_fitting

# Calculating df_median_slope_arrhenius
def calculate_median_slope_arrhenius(df_fitting):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+'arrhenius/'):
        os.makedirs('figures/'+folder_run_name+'arrhenius/')
        os.makedirs('figures/'+folder_run_name+'arrhenius_slope/')
    
    parameter_slope_list = ['cycle_PCEmppArea_norm_slope',
                            'cycle_VmppArea_norm_slope',
                            'time_PCEmppArea_norm_slope',
                            'time_VmppArea_norm_slope',
                            'cycle_PCEmppArea_slope',
                            'cycle_VmppArea_slope',
                            'time_PCEmppArea_slope', 
                            'time_VmppArea_slope',
                            # Starting from here: Loss
                            'cycle_PCEmppAreaLoss_norm_slope',
                            'cycle_VmppAreaLoss_norm_slope',
                            'time_PCEmppAreaLoss_norm_slope',
                            'time_VmppAreaLoss_norm_slope',
                            'cycle_PCEmppAreaLoss_slope',
                            'cycle_VmppAreaLoss_slope',
                            'time_PCEmppAreaLoss_slope',
                            'time_VmppAreaLoss_slope',
                            # Starting from here: max PCEmpp and max Vmpp
                            'cycle_maxPCEmpp_norm_slope',
                            'cycle_maxVmpp_norm_slope',
                            'time_maxPCEmpp_norm_slope',
                            'time_maxVmpp_norm_slope',
                            'cycle_maxPCEmpp_slope',
                            'cycle_maxVmpp_slope',
                            'time_maxPCEmpp_slope',
                            'time_maxVmpp_slope',
                           ]
    
    # batch_int='Ulas'
    # selected = df_fitting[df_fitting['device_name']==batch_int]
    # param = 'time_PCEmppArea_norm_slope'
    list_slope_median = []
    list_arrhenius_median = []
    list_device_name = []
    list_temperature = []
    list_oneperT = []
    list_parameter = []
    
    
    for device in batchname_list:
        
        selected = df_fitting[df_fitting['device_name']==device]
        
        # Go through each parameters
        for param in parameter_slope_list:
            
            print('device: ', device, ' param: ', param)
            
            # Temperature
            oneperT = np.ndarray.round(1/(selected['temperature'].unique()+273), decimals=5)
    
            # Slope plot
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            ax = sns.boxplot(x=1/(selected['temperature']+273), y=((selected[param])), palette="Blues")
            slope_medians = selected.groupby(['temperature'])[param].median()
    
            ax.set_xticklabels(np.flip(oneperT))#, rotation=45)
            ax.set_xlabel('1/temperature')
            plt.tight_layout()
    
            # Save figure
            plt.savefig('figures/'+folder_run_name+'arrhenius_slope/'+
                        device+'_'+param+'.png', dpi=600)
    
            plt.close()
    
            # Arrhenius plot
            selected['ln(slope)'] = np.log(np.abs(selected[param]))
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            ax = sns.boxplot(x=1/(selected['temperature']+273), y=selected['ln(slope)'], palette="Blues")
    
            arrhenius_medians = selected.groupby(['temperature'])['ln(slope)'].median()
    
            ax.set_xticklabels(np.flip(oneperT))#, rotation=45)
            ax.set_xlabel('1/temperature')
            plt.tight_layout()
    
            # Save figure
            plt.savefig('figures/'+folder_run_name+'arrhenius/'+
                        device+'_'+param+'.png', dpi=600)
    
            plt.close()
    
            # Append list slope medians and arrhenius medians
            slope_medians = list(np.array(slope_medians).flat)
            list_slope_median.append(slope_medians)#.to_list())
    
            arrhenius_medians = list(np.array(arrhenius_medians).flat)
            list_arrhenius_median.append(arrhenius_medians)
    
            list_device_name.append([device, device, device])
    
            list_temperature.append(df_fitting['temperature'].unique()+273)
            list_oneperT.append(oneperT)
    
            list_parameter.append([param,param,param])
    
    # Flatten
    list_slope_median = list(np.array(list_slope_median).flat)
    list_arrhenius_median = list(np.array(list_arrhenius_median).flat)
    list_device_name = list(np.array(list_device_name).flat)
    list_parameter = list(np.array(list_parameter).flat)
    
    list_oneperT = list(np.array(list_oneperT).flat)
    list_temperature = list(np.array(list_temperature).flat)
    
    # Create df
    df_median_slope_arrhenius = pd.DataFrame(list(zip(list_device_name,list_parameter,
                                                      list_temperature,list_oneperT,
                                                      list_slope_median,list_arrhenius_median)),
                                             columns=['device_name','parameter','t_K','1/T_1/K',
                                                      'slope_median','ln(slope)_median'])
    
    # Save df_median_slope_arrhenius
    df_median_slope_arrhenius.to_csv('output_dataframe/'+folder_run_name+
                                     'df_median_slope_arrhenius.csv',index=False)
    

# Calculating df_median_slope_arrhenius WITH UNIFORM YLIM
def calculate_median_slope_arrhenius_ylim(df_fitting):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+'arrhenius/ylim_uniform/'):
        os.makedirs('figures/'+folder_run_name+'arrhenius/ylim_uniform/')
        os.makedirs('figures/'+folder_run_name+'arrhenius_slope/ylim_uniform/')
    
    parameter_slope_list = ['cycle_PCEmppArea_norm_slope',
                            'cycle_VmppArea_norm_slope',
                            'time_PCEmppArea_norm_slope',
                            'time_VmppArea_norm_slope',
                            'cycle_PCEmppArea_slope',
                            'cycle_VmppArea_slope',
                            'time_PCEmppArea_slope', 
                            'time_VmppArea_slope',
                            # Starting from here: Loss
                            # 'cycle_PCEmppAreaLoss_norm_slope',
                            'cycle_VmppAreaLoss_norm_slope',
                            # 'time_PCEmppAreaLoss_norm_slope',
                            'time_VmppAreaLoss_norm_slope',
                            # 'cycle_PCEmppAreaLoss_slope',
                            'cycle_VmppAreaLoss_slope',
                            # 'time_PCEmppAreaLoss_slope',
                            'time_VmppAreaLoss_slope',
                            # Starting from here: max PCEmpp and max Vmpp
                            'cycle_maxPCEmpp_norm_slope',
                            # 'cycle_maxVmpp_norm_slope',
                            'time_maxPCEmpp_norm_slope',
                            # 'time_maxVmpp_norm_slope',
                            'cycle_maxPCEmpp_slope',
                            # 'cycle_maxVmpp_slope',
                            'time_maxPCEmpp_slope',
                            # 'time_maxVmpp_slope',
                           ]
    
    # batch_int='Ulas'
    # selected = df_fitting[df_fitting['device_name']==batch_int]
    # param = 'time_PCEmppArea_norm_slope'
    list_slope_median = []
    list_arrhenius_median = []
    list_device_name = []
    list_temperature = []
    list_oneperT = []
    list_parameter = []
    
    # Dictionary key for ylim
    ylim_dict = dict([
        ('cycle_PCEmppArea_norm_slope',[-0.003, 0.005]),
        ('cycle_PCEmppArea_slope',[-0.035, 0.08]),
        ('time_PCEmppArea_norm_slope',[-0.0006, 0.00025]),
        ('time_PCEmppArea_slope',[-0.0065, 0.0035]),
        ('cycle_VmppArea_norm_slope',[-0.0025, 0.005]),
        ('cycle_VmppArea_slope',[-0.002, 0.0045]),
        ('time_VmppArea_norm_slope',[-0.0004, 0.00025]),
        ('time_VmppArea_slope',[-0.0003, 0.0002]),
        ('cycle_maxPCEmpp_norm_slope',[-0.0025, 0.005]),
        ('cycle_maxPCEmpp_slope',[-0.035, 0.08]),
        ('time_maxPCEmpp_norm_slope',[-0.0005, 0.00025]),
        ('time_maxPCEmpp_slope',[-0.008, 0.004]),
        ('cycle_VmppAreaLoss_norm_slope',[-0.012, 0.002]),
        ('cycle_VmppAreaLoss_slope',[-0.0012, 0.0002]),
        ('time_VmppAreaLoss_norm_slope',[-0.0005, 0.0002]),
        ('time_VmppAreaLoss_slope',[-0.00006, 0.00003]),
    ])
    
    
    for device in batchname_list:
        
        selected = df_fitting[df_fitting['device_name']==device]
        
        # Go through each parameters
        for param in parameter_slope_list:
            
            print('device: ', device, ' param: ', param)
            
            # Temperature
            oneperT = np.ndarray.round(1/(selected['temperature'].unique()+273), decimals=5)
    
            # Slope plot
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            ax = sns.boxplot(x=1/(selected['temperature']+273), y=((selected[param])), palette="Blues")
            slope_medians = selected.groupby(['temperature'])[param].median()
    
            ax.set_xticklabels(np.flip(oneperT))#, rotation=45)
            ax.set_xlabel('1/temperature')
            ax.set_ylim(ylim_dict[param])
            plt.tight_layout()
    
            # Save figure
            plt.savefig('figures/'+folder_run_name+'arrhenius_slope/ylim_uniform/'+
                        device+'_'+param+'.png', dpi=600)
    
            plt.close()
    
            # Arrhenius plot
            selected['ln(slope)'] = np.log(np.abs(selected[param]))
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
            ax = sns.boxplot(x=1/(selected['temperature']+273), y=selected['ln(slope)'], palette="Blues")
    
            arrhenius_medians = selected.groupby(['temperature'])['ln(slope)'].median()
    
            ax.set_xticklabels(np.flip(oneperT))#, rotation=45)
            ax.set_xlabel('1/temperature')
            ax.set_ylim(ylim_dict[param])
            plt.tight_layout()
    
            # Save figure
            plt.savefig('figures/'+folder_run_name+'arrhenius/ylim_uniform/'+
                        device+'_'+param+'.png', dpi=600)
    
            plt.close()
    
            # Append list slope medians and arrhenius medians
            slope_medians = list(np.array(slope_medians).flat)
            list_slope_median.append(slope_medians)#.to_list())
    
            arrhenius_medians = list(np.array(arrhenius_medians).flat)
            list_arrhenius_median.append(arrhenius_medians)
    
            list_device_name.append([device, device, device])
    
            list_temperature.append(df_fitting['temperature'].unique()+273)
            list_oneperT.append(oneperT)
    
            list_parameter.append([param,param,param])
    
    # Flatten
    list_slope_median = list(np.array(list_slope_median).flat)
    list_arrhenius_median = list(np.array(list_arrhenius_median).flat)
    list_device_name = list(np.array(list_device_name).flat)
    list_parameter = list(np.array(list_parameter).flat)
    
    list_oneperT = list(np.array(list_oneperT).flat)
    list_temperature = list(np.array(list_temperature).flat)
    
    # Create df
    df_median_slope_arrhenius = pd.DataFrame(list(zip(list_device_name,list_parameter,
                                                      list_temperature,list_oneperT,
                                                      list_slope_median,list_arrhenius_median)),
                                             columns=['device_name','parameter','t_K','1/T_1/K',
                                                      'slope_median','ln(slope)_median'])

# Performing regression on the arrhenius
def regression_arrhenius(df_fitting):    
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+'arrhenius_fitting/'):
        os.makedirs('figures/'+folder_run_name+'arrhenius_fitting/')
    
    # Empty lists
    list_device_name = []
    list_parameter = []
    list_slope_regression_Arrhenius = []
    list_intercept_regression_Arrhenius = []
    
    # Parameter slope list
    parameter_slope_list = ['cycle_PCEmppArea_norm_slope',
                        'cycle_VmppArea_norm_slope',
                        'time_PCEmppArea_norm_slope',
                        'time_VmppArea_norm_slope',
                        'cycle_PCEmppArea_slope',
                        'cycle_VmppArea_slope',
                        'time_PCEmppArea_slope', 
                        'time_VmppArea_slope',
                        # Starting from here: Loss
#                         'cycle_PCEmppAreaLoss_norm_slope',
                        'cycle_VmppAreaLoss_norm_slope',
#                         'time_PCEmppAreaLoss_norm_slope',
                        'time_VmppAreaLoss_norm_slope',
#                         'cycle_PCEmppAreaLoss_slope',
                        'cycle_VmppAreaLoss_slope',
#                         'time_PCEmppAreaLoss_slope',
                        'time_VmppAreaLoss_slope',
                        # Starting from here: max PCEmpp and max Vmpp
                        'cycle_maxPCEmpp_norm_slope',
#                         'cycle_maxVmpp_norm_slope',
                        'time_maxPCEmpp_norm_slope',
#                         'time_maxVmpp_norm_slope',
                        'cycle_maxPCEmpp_slope',
#                         'cycle_maxVmpp_slope',
                        'time_maxPCEmpp_slope',
#                         'time_maxVmpp_slope',
                       ]
    
   
    for device in batchname_list:
    
        selected = df_fitting[df_fitting['device_name']==device]
        
        # Go through each parameters
        for param in parameter_slope_list:
    
    # selected = df_fitting[df_fitting['device_name']==device]
    
            # Arrhenius
            selected['ln(slope)'] = np.log(np.abs(selected[param]))
            selected['1/T_1/K'] = 1/(selected['temperature']+273)
            sns.set(rc={'figure.figsize':(5,3)})
            plt.figure()
    
    
            # Plot the overview and OLS trendline
            ax = sns.regplot(data=selected, x='1/T_1/K', y='ln(slope)',
                             fit_reg=True,ci = 99,
                             scatter_kws={'alpha':0.3}, # "color": "blue",
                             line_kws={"color": "red"})#,'alpha':0.8})
       
            # #calculate slope and intercept of regression equation
            slope, intercept, r, p, sterr = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                                   y=ax.get_lines()[0].get_ydata())
    
            # ax.set_xticklabels(np.flip(oneperT))#, rotation=45)
            # ax.set_xlabel('1/temperature')
            plt.tight_layout()
    
            # # Save figure
            plt.savefig('figures/'+folder_run_name+'arrhenius_fitting/'+device+'_'+param+'.png', dpi=600)
            
            # Close figure
            plt.close()
            
            # Append list
            list_device_name.append(device)
            list_parameter.append(param)
            list_slope_regression_Arrhenius.append(slope)
            list_intercept_regression_Arrhenius.append(intercept)
            
    # Create df
    df_regression_arrhenius = pd.DataFrame(list(zip(list_device_name,list_parameter,
                                                    list_slope_regression_Arrhenius,
                                                    list_intercept_regression_Arrhenius)),
                                           columns=['device_name','parameter',
                                                    'slope_regression_Arrhenius',
                                                    'intercept_regression_Arrhenius'])
    
    # Calculate Ea/ activation energy, and R = 8.3145 J mol^-1 K^-1
    R_constant = 8.3145
    c_Jmol_to_eVpar = 1.0364E-5
    df_regression_arrhenius['Ea_J_mol-1'] = -R_constant*df_regression_arrhenius['slope_regression_Arrhenius']
    df_regression_arrhenius['Ea_eV_perParticle'] = c_Jmol_to_eVpar*df_regression_arrhenius['Ea_J_mol-1']
    df_regression_arrhenius['A'] = np.exp(df_regression_arrhenius['intercept_regression_Arrhenius'])
    
    # Save df
    df_regression_arrhenius.to_csv('figures/'+folder_run_name+'arrhenius_fitting/'+
                                   'df_regression_arrhenius.csv',index=False)

    
    return df_regression_arrhenius

###############################################################################
# FUNCTIONS: FUNCTIONS FOR FITTING CYCLIC AGING TEST
###############################################################################

# Define the logarithmic function
def logarithmic_func(x, a, b, c):
    return a * np.log(b * x) + c

# Define logarithmic fit
def logarithmic_fit(x, y):
    
    # Remove non-positive values from the data
    mask = y > 0
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [0.06,0.5,1]
    
    # Perform the curve fitting
    popt, _ = curve_fit(logarithmic_func, x_fit, y_fit,
                        maxfev=100000, p0 = initial_guess)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit = popt
#     a_fit = popt

    return a_fit, b_fit, c_fit

# Define the logarithmic function + linear
def logarithmic_linear_func(x, a, b, c, d):
    return a * np.log(b * x) + c*x + d

# Define logarithmic fit + linear
def logarithmic_linear_fit(x, y):
    
    # Remove non-positive values from the data
    mask = y > 0
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [0.06,0.5,5,1]
    
    # Perform the curve fitting
    popt, _ = curve_fit(logarithmic_linear_func, x_fit, y_fit,
                        maxfev=100000, p0 = initial_guess)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit, d_fit = popt
#     a_fit = popt

    return a_fit, b_fit, c_fit, d_fit

# Define the power function
def power_func(x,a,b):
    return a * x**b

# Define the power function fit
def power_fit(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [1.0,1.0]
    
    # Perform the curve fitting
    popt, _ = curve_fit(power_func, x, y, p0=initial_guess)

    # Extract the fitted parameters
    a_fit, b_fit = popt

    return a_fit, b_fit#, c_fit


# Define the exponential function
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Define the exp function fit
def exp_fit(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [1.0,1.0,1.0]
    
    # Perform the curve fitting
    popt, _ = curve_fit(exp_func, x, y, p0=initial_guess,maxfev=5000)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit = popt

    return a_fit, b_fit, c_fit

# Define the exponential function
def exp_linear_func(x, a, b, c, d):
    return a * np.exp(-b * x) + c*x + d

# Define the exp function fit
def exp_linear_fit(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [1.0,1.0,1.0,1.0]
    
    # Perform the curve fitting
    popt, _ = curve_fit(exp_linear_func, x, y, p0=initial_guess,maxfev=100000)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit, d_fit = popt

    return a_fit, b_fit, c_fit, d_fit

# Define the tanh function
def tanh_func(x,a,b,c):
    return a * np.tanh(b * x - c)

# Define the tanh function fit
def tanh_fit(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [1.0,0.01,0.0]
    
    # Perform the curve fitting
    popt, _ = curve_fit(tanh_func, x, y, p0=initial_guess,maxfev=100000)

    # Extract the fitted parameters
    a_fit, b_fit,c_fit = popt

    return a_fit, b_fit,c_fit

def r_squared(y_observed, y_predicted):
    """
    Calculate the R-squared value for curve fitting.

    Parameters:
        y_observed (numpy array): Array of observed data points.
        y_predicted (numpy array): Array of predicted/fitted values.

    Returns:
        float: R-squared value.
    """
    # Calculate the mean of the observed data
    y_mean = np.mean(y_observed)

    # Calculate the sum of squares of residuals (SSR)
    ssr = np.sum((y_observed - y_predicted) ** 2)

    # Calculate the total sum of squares (SST)
    sst = np.sum((y_observed - y_mean) ** 2)

    # Calculate the R-squared value
    r_squared = 1 - (ssr / sst)

    return r_squared

#### Integral of each function
# Integral of the logarithmic_func for time 0 until time = time
def integral_logarithmic_func(time, a, b, c):
    return time * (a * np.log(b * time) - a + c)

# Integral of the logarithmic_linear_func for time 0 until time = time
def integral_logarithmic_linear_func(time, a, b, c, d):
    return time * (a * np.log(b * time) - a + c*time/2 + d)

# Integral of the power_func for time 0 until time = time
def integral_power_func(time, a, b):
    return (a * time**(b+1))/(b+1)

# Integral of the exp_func for time 0 until time = time
def integral_exp_func(time, a, b, c):
    return (c*time) - (a * np.exp(-b * time))/b + (a/b) # last term is time = 0

# Integral of the exp_linear_func for time 0 until time = time
def integral_exp_linear_func(time, a, b, c, d):
    return (c*time**2)/2 + d*time - (a * np.exp(-b * time))/b + (a/b) # last term is time = 0

# Integral of the tanh_func for time 0 until time = time
def integral_tanh_func(time, a, b, c):
    return (a * np.log(np.cosh(c-b*time)))/b - (a * np.log(np.cosh(c)))/b # last term is time = 0

#### Area differences
def area_difference_log(x, target_area, a, b, c):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(logarithmic_func, 0, x, args=(a, b, c))
    return abs(area - target_area)

def area_difference_log_linear(x, target_area, a, b, c, d):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(logarithmic_linear_func, 0, x, args=(a, b, c, d))
    return abs(area - target_area)

def area_difference_power(x, target_area, a, b):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(power_func, 0, x, args=(a, b))
    return abs(area - target_area)

def area_difference_exp(x, target_area, a, b, c):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(exp_func, 0, x, args=(a, b, c))
    return abs(area - target_area)

def area_difference_exp_linear(x, target_area, a, b, c, d):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(exp_linear_func, 0, x, args=(a, b, c, d))
    return abs(area - target_area)

def area_difference_tanh(x, target_area, a, b, c):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(tanh_func, 0, x, args=(a, b, c))
    return abs(area - target_area)


##### Not normalized
# Define logarithmic fit
def logarithmic_fit_notNorm(x, y):
    
    # Remove non-positive values from the data
    mask = y > 0
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [0.5,0.5,10]
    
    # Perform the curve fitting
    popt, _ = curve_fit(logarithmic_func, x_fit, y_fit,
                        maxfev=100000, p0 = initial_guess)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit = popt
#     a_fit = popt

    return a_fit, b_fit, c_fit

# Define logarithmic linear fit
def logarithmic_linear_fit_notNorm(x, y):
    
    # Remove non-positive values from the data
    mask = y > 0
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [0.8,0.8,-0.4,10]
    
    # Perform the curve fitting
    popt, _ = curve_fit(logarithmic_linear_func, x_fit, y_fit,
                        maxfev=100000, p0 = initial_guess)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit, d_fit = popt
#     a_fit = popt

    return a_fit, b_fit, c_fit, d_fit

# Define the exp function fit
def exp_fit_notNorm(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [-3,4,10]
    
    # Perform the curve fitting
    popt, _ = curve_fit(exp_func, x, y, p0=initial_guess,maxfev=100000)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit = popt

    return a_fit, b_fit, c_fit

# Define the exp function fit
def exp_linear_fit_notNorm(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [-5,7,0.05,10]
    
    # Perform the curve fitting
    popt, _ = curve_fit(exp_linear_func, x, y, p0=initial_guess,maxfev=100000)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit, d_fit = popt

    return a_fit, b_fit, c_fit, d_fit

#### Area differences not norm
def area_difference_log_notNorm(x, target_area, a, b, c):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(logarithmic_func, 0, x, args=(a, b, c))
    return abs(area/x - target_area)

def area_difference_log_linear_notNorm(x, target_area, a, b, c, d):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(logarithmic_linear_func, 0, x, args=(a, b, c, d))
    return abs(area/x - target_area)

def area_difference_power_notNorm(x, target_area, a, b):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(power_func, 0, x, args=(a, b))
    return abs(area/x - target_area)

def area_difference_exp_notNorm(x, target_area, a, b, c):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(exp_func, 0, x, args=(a, b, c))
    return abs(area/x - target_area)

def area_difference_exp_linear_notNorm(x, target_area, a, b, c, d):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(exp_linear_func, 0, x, args=(a, b, c, d))
    return abs(area/x - target_area)

def area_difference_tanh_notNorm(x, target_area, a, b, c):
    # Integrate the function from 0 to x to get the area
    area, _ = quad(tanh_func, 0, x, args=(a, b, c))
    return abs(area/x - target_area)


###############################################################################
# FUNCTIONS: FINDING COMPARABLE AREA ACROSS DIFFERENT T FOR SPECIFIC BATCH
###############################################################################

def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

def flatten_list(input_list):
    flattened = []
    for item in input_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def calculate_median_specific_T(df_batch_selected_xC, df_median_xC):
    
    list_45C_median = []

    for time in df_median_xC['time_h']:
        list_45C = []
    #     print('time: ',time)
        
        for index, row in df_batch_selected_xC.iterrows():        
    #         print('index: ',index)
            selected_45C = df_batch_selected_xC['MPPT'][index]
            value_45C = (selected_45C[selected_45C['time_h']==time])['PCE_mpp'].tolist()
    #         print(value_55C)
            list_45C.append(value_45C)
            
        # Need to flatten list
        list_45C = flatten_comprehension(list_45C)
        median_45C = np.median(list_45C)
        list_45C_median.append(median_45C)
    
    df_median_xC['PCEmpp_median']=list_45C_median
        
    return df_median_xC


def fit_aging_cycle_function(df_all_35C, df_all_45C, df_all_55C):
    
    batchname_list_tracking = []
    
    # Go through each batch name
    for batch_name in batchname_list:
        
        # Select specific batch for each temperature
        df_batch_selected_35C = df_all_35C[(df_all_35C['temperature']==35) & (df_all_35C['device_name']==batch_name)]
        df_batch_selected_45C = df_all_45C[(df_all_45C['temperature']==45) & (df_all_45C['device_name']==batch_name)]
        df_batch_selected_55C = df_all_55C[(df_all_55C['temperature']==55) & (df_all_55C['device_name']==batch_name)]

        # Go through each row
        df_median_55C = ((df_batch_selected_55C['MPPT'].iloc[0])['time_h']).to_frame()
        df_median_45C = ((df_batch_selected_45C['MPPT'].iloc[0])['time_h']).to_frame()
        df_median_35C = ((df_batch_selected_35C['MPPT'].iloc[0])['time_h']).to_frame()
        
        df_median_55C['time_h_collapsed']=((df_batch_selected_55C['MPPT'].iloc[0])['time_h_collapsed'])
        df_median_45C['time_h_collapsed']=((df_batch_selected_45C['MPPT'].iloc[0])['time_h_collapsed'])
        df_median_35C['time_h_collapsed']=((df_batch_selected_35C['MPPT'].iloc[0])['time_h_collapsed'])
        
        df_median_55C['cycle']=((df_batch_selected_55C['MPPT'].iloc[0])['cycle'])
        df_median_45C['cycle']=((df_batch_selected_45C['MPPT'].iloc[0])['cycle'])
        df_median_35C['cycle']=((df_batch_selected_35C['MPPT'].iloc[0])['cycle'])

        df_median_55C = calculate_median_specific_T(df_batch_selected_55C, df_median_55C)
        df_median_45C = calculate_median_specific_T(df_batch_selected_45C, df_median_45C)
        df_median_35C = calculate_median_specific_T(df_batch_selected_35C, df_median_35C)
        
        # Need to divide by # max of the column? Yes
        max_value_35C = df_median_35C['PCEmpp_median'].max()
        max_value_45C = df_median_45C['PCEmpp_median'].max()
        max_value_55C = df_median_55C['PCEmpp_median'].max()
        
        df_median_35C['PCEmpp_median_norm'] = df_median_35C['PCEmpp_median']/max_value_35C
        df_median_45C['PCEmpp_median_norm'] = df_median_45C['PCEmpp_median']/max_value_45C
        df_median_55C['PCEmpp_median_norm'] = df_median_55C['PCEmpp_median']/max_value_55C
                
        # Fit all cycles
        df_fitting_cycle_35C = fit_all_cycles(df_median_35C, 35, batch_name)
        df_fitting_cycle_45C = fit_all_cycles(df_median_45C, 45, batch_name)
        df_fitting_cycle_55C = fit_all_cycles(df_median_55C, 55, batch_name)
        
        batchname_list_tracking.append([batch_name,df_fitting_cycle_35C,
                                        df_fitting_cycle_45C, df_fitting_cycle_55C])
        
    return batchname_list_tracking
        

def fit_all_cycles(df_median_45C, temperature, batch_name):
    
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+'cycle_fitting/'):
        os.makedirs('figures/'+folder_run_name+'cycle_fitting/')
    
    unique_45C_cycle = df_median_45C['cycle'].unique()

    list_a_fit_log = []
    list_b_fit_log = []
    list_c_fit_log = []
    list_r_squared_log = []
    
    list_a_fit_log_linear = []
    list_b_fit_log_linear = []
    list_c_fit_log_linear = []
    list_d_fit_log_linear = []
    list_r_squared_log_linear = []

    list_a_fit_exp = []
    list_b_fit_exp = []
    list_c_fit_exp = []
    list_r_squared_exp = []
    
    list_a_fit_exp_linear = []
    list_b_fit_exp_linear = []
    list_c_fit_exp_linear = []
    list_d_fit_exp_linear = []
    list_r_squared_exp_linear = []

    list_a_fit_power = []
    list_b_fit_power = []
    list_r_squared_power = []

    list_a_fit_tanh = []
    list_b_fit_tanh = []
    list_c_fit_tanh = []
    list_r_squared_tanh = []
    
    # If temperature == 55 remove certain cycle and certain batch
    if (temperature ==55) :#& (batch_name=='Zahra-Batch1'):
        
        # List of values to be dropped
        values_to_drop = [344, 345, 346] 

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
        
    elif (temperature ==45) & (batch_name=='Ulas'):
        
        # List of values to be dropped
        values_to_drop = [173] 

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
    
    elif (temperature ==45) & (batch_name=='Zahra-Batch2'):
        
        # List of values to be dropped
        values_to_drop = [173] 

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]

    for cycle in unique_45C_cycle:

        # cycle = 10
        # print(cycle)

        df_median_45C_cycleX = df_median_45C[df_median_45C['cycle']==cycle]
        df_median_45C_cycleX

        # Pick x_data and y_data
        x_data = df_median_45C_cycleX['time_h_collapsed']
        y_data = df_median_45C_cycleX['PCEmpp_median_norm']

        ##### LOGARITHMIC
        # Fit the data to the natural logarithmic equation
        a_fit_log, b_fit_log, c_fit_log = logarithmic_fit(x_data,y_data)
        list_a_fit_log.append(a_fit_log)
        list_b_fit_log.append(b_fit_log)
        list_c_fit_log.append(c_fit_log)

        # Generate points for the fitted curve
        y_fit_log = logarithmic_func(x_data, a_fit_log, b_fit_log, c_fit_log)
        r_squared_log =  r_squared(y_data, y_fit_log)
        list_r_squared_log.append(r_squared_log)
        
        ##### LOGARITHMIC LINEAR
        # Fit the data to the natural logarithmic linear equation
        a_fit_log_linear, b_fit_log_linear, c_fit_log_linear, d_fit_log_linear = logarithmic_linear_fit(x_data,y_data)
        list_a_fit_log_linear.append(a_fit_log_linear)
        list_b_fit_log_linear.append(b_fit_log_linear)
        list_c_fit_log_linear.append(c_fit_log_linear)
        list_d_fit_log_linear.append(d_fit_log_linear)

        # Generate points for the fitted curve
        y_fit_log_linear = logarithmic_linear_func(x_data, a_fit_log_linear, b_fit_log_linear,
                                                   c_fit_log_linear, d_fit_log_linear)
        r_squared_log_linear =  r_squared(y_data, y_fit_log_linear)
        list_r_squared_log_linear.append(r_squared_log_linear)

        ##### EXP DECAY UPWARDS
        # Fit the data to the exp decay upwards equation
        a_fit_exp, b_fit_exp, c_fit_exp = exp_fit(x_data,y_data)
        list_a_fit_exp.append(a_fit_exp)
        list_b_fit_exp.append(b_fit_exp)
        list_c_fit_exp.append(c_fit_exp)

        # Generate points for the fitted curve
        y_fit_exp = exp_func(x_data, a_fit_exp, b_fit_exp, c_fit_exp)
        r_squared_exp =  r_squared(y_data, y_fit_exp)
        list_r_squared_exp.append(r_squared_exp)
        
        ##### EXP DECAY UPWARDS LINEAR
        # Fit the data to the exp decay upwards linear equation
        a_fit_exp_linear, b_fit_exp_linear, c_fit_exp_linear, d_fit_exp_linear = exp_linear_fit(x_data,y_data)
        list_a_fit_exp_linear.append(a_fit_exp_linear)
        list_b_fit_exp_linear.append(b_fit_exp_linear)
        list_c_fit_exp_linear.append(c_fit_exp_linear)
        list_d_fit_exp_linear.append(d_fit_exp_linear)

        # Generate points for the fitted curve
        y_fit_exp_linear = exp_linear_func(x_data, a_fit_exp_linear, b_fit_exp_linear,
                                           c_fit_exp_linear, d_fit_exp_linear)
        r_squared_exp_linear =  r_squared(y_data, y_fit_exp_linear)
        list_r_squared_exp_linear.append(r_squared_exp_linear)
        
        ##### POWER
        # Fit the data to the power equation
        a_fit_power, b_fit_power = power_fit(x_data,y_data)
        list_a_fit_power.append(a_fit_power)
        list_b_fit_power.append(b_fit_power)

        # Generate points for the fitted curve
        y_fit_power = power_func(x_data, a_fit_power, b_fit_power)
        r_squared_power = r_squared(y_data, y_fit_power)
        list_r_squared_power.append(r_squared_power)

        ##### TANH 
        # Fit the data to the natural logarithmic equation
        a_fit_tanh, b_fit_tanh, c_fit_tanh = tanh_fit(x_data,y_data)
        list_a_fit_tanh.append(a_fit_tanh)
        list_b_fit_tanh.append(b_fit_tanh)
        list_c_fit_tanh.append(c_fit_tanh)

        # Generate points for the fitted curve
        y_fit_tanh = tanh_func(x_data, a_fit_tanh, b_fit_tanh, c_fit_tanh)
        r_squared_tanh =  r_squared(y_data, y_fit_tanh)
        list_r_squared_tanh.append(r_squared_tanh)

        ##### PLOT

        # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
        plt.close()
        plt.figure(figsize=(15, 8))

        # Subplot 1: Natural Logarithm
        plt.subplot(2, 3, 1)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_log, label='Fitted curve', color='red')
        plt.title(f'Fitted Natural Logarithmic Curve, r2: {r_squared_log:.3f}')
        # plt.legend()
        
        # Subplot 2: Natural Logarithm Linear
        plt.subplot(2, 3, 2)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_log_linear, label='Fitted curve', color='red')
        plt.title(f'Fitted Natural Logarithmic + Linear Curve, r2: {r_squared_log_linear:.3f}')
        # plt.legend()
        
        # Subplot 3: Power Function
        plt.subplot(2, 3, 3)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_power, label='Fitted curve', color='red')
        plt.title(f'Fitted Power Curve, r2: {r_squared_power:.3f}')
        # plt.legend()
        
        # Subplot 4: Exponential Decay Upward
        plt.subplot(2, 3, 4)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_exp, label='Fitted curve', color='red')
        plt.title(f'Fitted Exponential Decay Upwards Curve, r2: {r_squared_exp:.3f}')
        # plt.legend()
        
        # Subplot 5: Exponential Decay Upward Linear
        plt.subplot(2, 3, 5)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_exp_linear, label='Fitted curve', color='red')
        plt.title(f'Fitted Exponential Decay Upwards + Linear Curve, r2: {r_squared_exp_linear:.3f}')
        # plt.legend()

        # Subplot 6: Hyperbolic Tangent (tanh)
        plt.subplot(2, 3, 6)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_tanh, label='Fitted curve', color='red')
        plt.title(f'Fitted Tanh Curve, r2: {r_squared_tanh:.3f}')
        plt.legend()

        plt.tight_layout()
        # Ensure that the "figures/curve_fitting" folder exists
        os.makedirs("figures/20230801_curve_fitting", exist_ok=True)

        # Save the figure in the "figures/curve_fitting" folder
        plt.savefig('figures/'+folder_run_name+'cycle_fitting/'+batch_name+"_"+str(temperature)+"_cycle_"+str(cycle)+".png",
                    dpi=200)

    #     plt.show()

    # Create the DataFrame
    df_fitting_cycle = pd.DataFrame({
        'cycle':unique_45C_cycle,
        'a_fit_log': list_a_fit_log,
        'b_fit_log': list_b_fit_log,
        'c_fit_log': list_c_fit_log,
        'r_squared_log': list_r_squared_log,
        'a_fit_log_linear': list_a_fit_log_linear,
        'b_fit_log_linear': list_b_fit_log_linear,
        'c_fit_log_linear': list_c_fit_log_linear,
        'd_fit_log_linear': list_d_fit_log_linear,
        'r_squared_log_linear': list_r_squared_log_linear,
        'a_fit_exp': list_a_fit_exp,
        'b_fit_exp': list_b_fit_exp,
        'c_fit_exp': list_c_fit_exp,
        'r_squared_exp': list_r_squared_exp,
        'a_fit_exp_linear': list_a_fit_exp_linear,
        'b_fit_exp_linear': list_b_fit_exp_linear,
        'c_fit_exp_linear': list_c_fit_exp_linear,
        'd_fit_exp_linear': list_d_fit_exp_linear,
        'r_squared_exp_linear': list_r_squared_exp_linear,
        'a_fit_power': list_a_fit_power,
        'b_fit_power': list_b_fit_power,
        'r_squared_power': list_r_squared_power,
        'a_fit_tanh': list_a_fit_tanh,
        'b_fit_tanh': list_b_fit_tanh,
        'c_fit_tanh': list_c_fit_tanh,
        'r_squared_tanh': list_r_squared_tanh,
    })

    # Save the dataframe
    df_fitting_cycle.to_csv('figures/'+folder_run_name+'cycle_fitting/fittingLog_'+batch_name+"_"+
                            str(temperature)+'_cycle_'+str(cycle)+'.csv',index=False)

    return df_fitting_cycle
    
    
###############################################################################
# FUNCTIONS: BACK-CALCULATE FOR EQUIVALENT OF 45C IN 55C, ETC.
###############################################################################

# Back calculate

def calculate_equivalent_time(df_fitting_cycle_45, df_fitting_cycle_55,
                              temp_target, temp_base,
                              batch_name,desired_foldername): # Find equivalent in 45C, base 55C

    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+desired_foldername):
        os.makedirs('figures/'+folder_run_name+desired_foldername)

    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Find the length of the base
    # if (temp_base == 45) and (temp_target == 55):
    #     desired_length_base = 1.5 # 55C
    # elif (temp_base == 55) and (temp_target == 45):
    #     desired_length_base = 3 # 45C
    # elif temp_base == 35: 
    #     desired_length_base = 3 # 45C
    
    if (temp_target == 55):
        desired_length_base = 1.5 # 55C
    elif (temp_target == 45):
        desired_length_base = 3 # 45C
    elif (temp_target == 35): 
        desired_length_base = 6 # 45C
    

    list_equivalent_45_time_log = []
    list_equivalent_45_time_log_linear = []
    list_equivalent_45_time_power = []
    list_equivalent_45_time_exp = []
    list_equivalent_45_time_exp_linear = []
    list_equivalent_45_time_tanh = []

    list_equivalent_45_diffArea_log = []
    list_equivalent_45_diffArea_log_linear = []
    list_equivalent_45_diffArea_power = []
    list_equivalent_45_diffArea_exp = []
    list_equivalent_45_diffArea_exp_linear = []
    list_equivalent_45_diffArea_tanh = []
    
    # Create a new list containing elements present in both lists (unique cycles)
    common_cycle = [x for x in unique_45C_cycle if x in unique_55C_cycle]
    

    # Go through each cycle
    for cycle in unique_45C_cycle:
        
        # If cycle is not on the common cycle:
        if cycle not in common_cycle:
            
            # Save the values
            list_equivalent_45_time_log.append(np.nan)
            list_equivalent_45_diffArea_log.append(np.nan)
            
            list_equivalent_45_time_log_linear.append(np.nan)
            list_equivalent_45_diffArea_log_linear.append(np.nan)
            
            list_equivalent_45_time_power.append(np.nan)
            list_equivalent_45_diffArea_power.append(np.nan)
            
            list_equivalent_45_time_exp.append(np.nan)
            list_equivalent_45_diffArea_exp.append(np.nan)
            
            list_equivalent_45_time_exp_linear.append(np.nan)
            list_equivalent_45_diffArea_exp_linear.append(np.nan)
            
            list_equivalent_45_time_tanh.append(np.nan)
            list_equivalent_45_diffArea_tanh.append(np.nan)
        
        # If cycle is common:
        elif cycle in common_cycle: 
            ### LOG
            # Parameters of the logarithmic function (replace with your values)
            a_55_log = df_fitting_cycle_55['a_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log = df_fitting_cycle_55['b_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log = df_fitting_cycle_55['c_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log = df_fitting_cycle_45['a_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log = df_fitting_cycle_45['b_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log = df_fitting_cycle_45['c_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log (target), based on the base 55C
            desired_area_45_log = integral_logarithmic_func(desired_length_base, a_55_log, b_55_log, c_55_log)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' log
            result_log = minimize_scalar(area_difference_log, args=(desired_area_45_log, a_45_log, b_45_log, c_45_log), method='bounded', bounds=(-10.0, 10.0))
            x_value_log = result_log.x

            # Desired area for 55
            area_45_log = integral_logarithmic_func(x_value_log, a_45_log, b_45_log, c_45_log)
            diff_45_log = area_45_log-desired_area_45_log

            # Save the values
            list_equivalent_45_time_log.append(x_value_log)
            list_equivalent_45_diffArea_log.append(diff_45_log)

            ### LOG + LINEAR
            # Parameters of the logarithmic function (replace with your values)
            a_55_log_linear = df_fitting_cycle_55['a_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log_linear = df_fitting_cycle_55['b_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log_linear = df_fitting_cycle_55['c_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_log_linear = df_fitting_cycle_55['d_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log_linear = df_fitting_cycle_45['a_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log_linear = df_fitting_cycle_45['b_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log_linear = df_fitting_cycle_45['c_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_log_linear = df_fitting_cycle_45['d_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log
            desired_area_45_log_linear = integral_logarithmic_linear_func(desired_length_base, a_55_log_linear, b_55_log_linear, c_55_log_linear, d_55_log_linear)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' log
            result_log_linear = minimize_scalar(area_difference_log_linear, args=(desired_area_45_log_linear, 
                                                                                  a_45_log_linear, b_45_log_linear,
                                                                                  c_45_log_linear, d_45_log_linear), method='bounded', bounds=(-10.0, 10.0))
            x_value_log_linear = result_log_linear.x

            # Desired area for 55
            area_45_log_linear = integral_logarithmic_linear_func(x_value_log_linear, a_45_log_linear, b_45_log_linear, c_45_log_linear, d_45_log_linear)
            diff_45_log_linear = area_45_log_linear-desired_area_45_log_linear

            # Save the values
            list_equivalent_45_time_log_linear.append(x_value_log_linear)
            list_equivalent_45_diffArea_log_linear.append(diff_45_log_linear)

            ### POWER
            # Parameters of the power function (replace with your values)
            a_55_power = df_fitting_cycle_55['a_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_power = df_fitting_cycle_55['b_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_power = df_fitting_cycle_45['a_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_power = df_fitting_cycle_45['b_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 power
            desired_area_45_power = integral_power_func(desired_length_base, a_55_power, b_55_power)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' power
            result_power = minimize_scalar(area_difference_power, args=(desired_area_45_power, a_45_power, b_45_power), method='bounded', bounds=(-10.0, 10.0))
            x_value_power = result_power.x

            # Desired area for 55
            area_45_power = integral_power_func(x_value_power, a_45_power, b_45_power)
            diff_45_power = area_45_power-desired_area_45_power

            # Save the values
            list_equivalent_45_time_power.append(x_value_power)
            list_equivalent_45_diffArea_power.append(diff_45_power)

            ### EXP
            # Parameters of the exp function (replace with your values)
            a_55_exp = df_fitting_cycle_55['a_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp = df_fitting_cycle_55['b_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp = df_fitting_cycle_55['c_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp = df_fitting_cycle_45['a_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp = df_fitting_cycle_45['b_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp = df_fitting_cycle_45['c_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
            desired_area_45_exp = integral_exp_func(desired_length_base, a_55_exp, b_55_exp, c_55_exp)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp = minimize_scalar(area_difference_exp, args=(desired_area_45_exp, a_45_exp, b_45_exp, c_45_exp), method='bounded', bounds=(-10.0, 10.0))
            x_value_exp = result_exp.x

            # Desired area for 55
            area_45_exp = integral_exp_func(x_value_exp, a_45_exp, b_45_exp, c_45_exp)
            diff_45_exp = area_45_exp-desired_area_45_exp

            # Save the values
            list_equivalent_45_time_exp.append(x_value_exp)
            list_equivalent_45_diffArea_exp.append(diff_45_exp)

            ### EXP + LINEAR
            # Parameters of the exp function (replace with your values)
            a_55_exp_linear = df_fitting_cycle_55['a_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp_linear = df_fitting_cycle_55['b_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp_linear = df_fitting_cycle_55['c_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_exp_linear = df_fitting_cycle_55['d_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp_linear = df_fitting_cycle_45['a_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp_linear = df_fitting_cycle_45['b_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp_linear = df_fitting_cycle_45['c_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_exp_linear = df_fitting_cycle_45['d_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
            desired_area_45_exp_linear = integral_exp_linear_func(desired_length_base, a_55_exp_linear, b_55_exp_linear, c_55_exp_linear, d_55_exp_linear)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp_linear = minimize_scalar(area_difference_exp_linear, args=(desired_area_45_exp_linear, a_45_exp_linear,
                                                                                  b_45_exp_linear, c_45_exp_linear,
                                                                                  d_45_exp_linear), method='bounded', bounds=(-10.0, 10.0))
            x_value_exp_linear = result_exp_linear.x

            # Desired area for 55
            area_45_exp_linear = integral_exp_linear_func(x_value_exp_linear, a_45_exp_linear, b_45_exp_linear, c_45_exp_linear, d_45_exp_linear)
            diff_45_exp_linear = area_45_exp_linear-desired_area_45_exp_linear

            # Save the values
            list_equivalent_45_time_exp_linear.append(x_value_exp_linear)
            list_equivalent_45_diffArea_exp_linear.append(diff_45_exp_linear)
            
            ### TANH
            # Parameters of the tanh function (replace with your values)
            a_55_tanh = df_fitting_cycle_55['a_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_tanh = df_fitting_cycle_55['b_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_tanh = df_fitting_cycle_55['c_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_tanh = df_fitting_cycle_45['a_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_tanh = df_fitting_cycle_45['b_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_tanh = df_fitting_cycle_45['c_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 tanh
            desired_area_45_tanh = integral_tanh_func(desired_length_base, a_55_tanh, b_55_tanh, c_55_tanh)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' tanh
            result_tanh = minimize_scalar(area_difference_tanh, args=(desired_area_45_tanh, a_45_tanh, b_45_tanh, c_45_tanh), method='bounded', bounds=(-10.0, 10.0))
            x_value_tanh = result_tanh.x

            # Desired area for 55
            area_45_tanh = integral_exp_func(x_value_tanh, a_45_tanh, b_45_tanh, c_45_tanh)
            diff_45_tanh = area_45_tanh-desired_area_45_tanh

            # Save the values
            list_equivalent_45_time_tanh.append(x_value_tanh)
            list_equivalent_45_diffArea_tanh.append(diff_45_tanh)
    
    # Prepare for the column name
    str_eq = 'eq_'+str(temp_target)+'C_wrt_'+str(temp_base)+'C_'
    str_eq_time = str_eq + 'time_'
    str_eq_diffArea = str_eq + 'diffArea_'
    list_func = ['log','linear_log','power','exp','linear_exp','tanh']
    
    str_time_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_time]*6, list_func)]
    str_diffArea_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_diffArea]*6, list_func)]
    str_all = str_time_all+ str_diffArea_all

    # Add to the df
    df_fitting_cycle_45[str_all[0]] = list_equivalent_45_time_log
    df_fitting_cycle_45[str_all[1]] = list_equivalent_45_time_log_linear
    df_fitting_cycle_45[str_all[2]] = list_equivalent_45_time_power
    df_fitting_cycle_45[str_all[3]] = list_equivalent_45_time_exp
    df_fitting_cycle_45[str_all[4]] = list_equivalent_45_time_exp_linear
    df_fitting_cycle_45[str_all[5]] = list_equivalent_45_time_tanh

    df_fitting_cycle_45[str_all[4]] = list_equivalent_45_diffArea_log
    df_fitting_cycle_45[str_all[5]] = list_equivalent_45_diffArea_log_linear
    df_fitting_cycle_45[str_all[6]] = list_equivalent_45_diffArea_power
    df_fitting_cycle_45[str_all[7]] = list_equivalent_45_diffArea_exp
    df_fitting_cycle_45[str_all[8]] = list_equivalent_45_diffArea_exp_linear
    df_fitting_cycle_45[str_all[9]] = list_equivalent_45_diffArea_tanh
    
    # Save the dataframe
    df_fitting_cycle_45.to_csv('figures/'+folder_run_name+desired_foldername+'fitting log_'+
                               batch_name+"_eq_Ttarget_"+str(temp_target)+'C_wrt_Tbase_'+
                               str(temp_base)+'_cycle_'+str(cycle)+'.csv',index=False)
    
    ### PLOT
    str_eq = str(temp_target)+'C wrt '+str(temp_base)+'C'
    x_data = df_fitting_cycle_45['cycle']
    y_data_log = df_fitting_cycle_45.filter(like='_time_log').iloc[:,0]
    y_data_log_linear = df_fitting_cycle_45.filter(like='_time_linear_log').iloc[:,0]
    y_data_power = df_fitting_cycle_45.filter(like='_time_power').iloc[:,0]
    y_data_exp = df_fitting_cycle_45.filter(like='_time_exp').iloc[:,0]
    y_data_exp_linear = df_fitting_cycle_45.filter(like='_time_linear_exp').iloc[:,0]
    y_data_tanh = df_fitting_cycle_45.filter(like='_time_tanh').iloc[:,0]

    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=y_data_log, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=y_data_log_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    plt.ylim([-5,8])

    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=y_data_power, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 4: Exponential Decay Upward
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=y_data_exp, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 5: Exponential Decay Upward + Linear
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=y_data_exp_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    plt.ylim([-5,8])

    # Subplot 6: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=y_data_tanh, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    plt.ylim([-5,8])

    plt.tight_layout()

    # plt.show()
    
    # Name txt file
    file_name_txt = 'figures/'+folder_run_name+desired_foldername+'recap_'+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt, "w") as file:        

        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log: {y_data_log.median()} ')
        file.write(f'Median log linear: {y_data_log_linear.median()} ')
        file.write(f'Median exp: {y_data_exp.median()} ')
        file.write(f'Median exp linear: {y_data_exp_linear.median()} ')
        file.write(f'Median power: {y_data_power.median()} ')
        file.write(f'Median tanh: {y_data_tanh.median()} ')

    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig('figures/'+folder_run_name+desired_foldername+batch_name+"_"+str_eq+".png",
                dpi=200)

    return df_fitting_cycle_45


###############################################################################
# FUNCTIONS: BACK-CALCULATE FOR EQUIVALENT OF 45C IN 55C, ETC. NOT NORMALIZED
###############################################################################

def area_under_PCEmpp_median_points(x, y):
    """
    Calculate the area under the scattered data points using Simpson's rule.

    Parameters:
        x: numpy array or list
            The x-coordinates of the scattered points.
        y: numpy array or list
            The y-coordinates of the scattered points.

    Returns:
        float
            The area under the scattered data points.
    """
    # Convert x and y to numpy arrays if provided as lists
    x = np.array(x)
    y = np.array(y)

    # Sort x and y based on x
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Calculate the area using Simpson's rule
    area = simps(y_sorted, x_sorted)

    return area


def create_df_area_under_PCEmpp_median(df_median):
    """
    Construct the dataframe under the PCEmpp_median points.
    
    Parameters:
        df_median : dataframe
            Dataframe containing the PCEmpp_median for 
            all cycles, specific temperature.
    
    Returns:
        dataframe
            Containing the area for each cycle.
    """
    # Find all the unique cycles
    unique_cycle = df_median['cycle'].unique()
    
    # Create empty list for storing the area
    list_area = []
    list_length_hour = []
    
    # Going through the loop
    for cycle in unique_cycle:
        
        # Select the dataframe
        selected = df_median[df_median['cycle']==cycle]
        
        # Calculate the delta time
        length_hour = selected['time_h'].iloc[-1]-selected['time_h'].iloc[0]
        
        # Define x and y
        x = selected['time_h_collapsed']
        y = selected['PCEmpp_median']
        
        # Calculate the area
        area = area_under_PCEmpp_median_points(x, y)
        
        # Append the value to the list
        list_area.append(area)
        list_length_hour.append(length_hour)
    
    # Calculated area dict
    calculated_area = {
        'cycle': unique_cycle,
        'PCEmppArea_median': list_area,
        'length_hour_perCycle': list_length_hour,
    }
    
    # Generate df
    df_calculated_area = pd.DataFrame(calculated_area)
    
    # Calculate the PCEmppArea_median_perHour
    df_calculated_area['PCEmppArea_median_perHour']=df_calculated_area['PCEmppArea_median']/df_calculated_area['length_hour_perCycle']
    
    return df_calculated_area

def fit_all_cycles_notNorm(df_median_45C, temperature, batch_name):
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+'cycle_fitting_notNorm/'):
        os.makedirs('figures/'+folder_run_name+'cycle_fitting_notNorm/')
    
    unique_45C_cycle = df_median_45C['cycle'].unique()

    list_a_fit_log = []
    list_b_fit_log = []
    list_c_fit_log = []
    list_r_squared_log = []
    
    list_a_fit_log_linear = []
    list_b_fit_log_linear = []
    list_c_fit_log_linear = []
    list_d_fit_log_linear = []
    list_r_squared_log_linear = []

    list_a_fit_exp = []
    list_b_fit_exp = []
    list_c_fit_exp = []
    list_r_squared_exp = []
    
    list_a_fit_exp_linear = []
    list_b_fit_exp_linear = []
    list_c_fit_exp_linear = []
    list_d_fit_exp_linear = []
    list_r_squared_exp_linear = []

    list_a_fit_power = []
    list_b_fit_power = []
    list_r_squared_power = []

    list_a_fit_tanh = []
    list_b_fit_tanh = []
    list_c_fit_tanh = []
    list_r_squared_tanh = []
    
    # If temperature == 55 remove certain cycle and certain batch
    if (temperature ==55) & (batch_name=='Zahra-Batch1'):
        
        # List of values to be dropped
        values_to_drop = [344, 345, 346,]
                          # 6,10,12,13,16,22,31,37,41,46,174,187,198,222,241,243,
                          # 244,250,252,260,264,277,283,297,321,349,351,
                          # ] # From other than error in linear log

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
        
    elif (temperature ==35) & (batch_name=='Zahra-Batch1'):
        
        # List of values to be dropped
        # values_to_drop = [2,3,4,5,6,8,9,13,14,17,21,22,24,33,41,42,63,67,78,
        #                   84,85,86,87,88,90,93,95,96,98,99,101,104,105,106,108,
        #                   113,114,118,120,135,154] 
        # values_to_drop = [2,5,12,15,]
        values_to_drop = [1,13,20,21]#cycle not looking good

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
        
    elif (temperature ==45) & (batch_name=='Zahra-Batch1'):
        
        # List of values to be dropped
        # values_to_drop = [2,3,4,5,6,8,9,13,14,17,21,22,24,33,41,42,63,67,78,
        #                   84,85,86,87,88,90,93,95,96,98,99,101,104,105,106,108,
        #                   113,114,118,120,135,154] 
        values_to_drop = [45,173]

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
        
    elif (temperature ==45) & (batch_name=='Zahra-Batch2'):
        
        # List of values to be dropped
        values_to_drop = [173,
                          53]  #cycle not looking good

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
        
    elif (temperature ==55) & (batch_name=='Zahra-Batch2'):
        
        # List of values to be dropped
        # values_to_drop = [2,4,6,8,17,20,26,30,32,34,35,36,39,40,41,42,46,47,58,61] 
        values_to_drop = [342, 344, 345]#, 346, 349, 351, 352]

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
       
    elif (temperature ==35) & (batch_name=='Zahra-Batch2'):
        
        # List of values to be dropped
        values_to_drop = [20,21,] #cycle not looking good 

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask] 
    
    elif (temperature ==35) & (batch_name=='Ulas'):
        
        # List of values to be dropped
        values_to_drop = [11,
                          20,21,] #cycle not looking good 

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]    
    
    elif (temperature ==45) & (batch_name=='Ulas'):
        
        # List of values to be dropped
        values_to_drop = [173]

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask]
    
    elif (temperature ==55) & (batch_name=='Ulas'):
        
        # List of values to be dropped
        # # values_to_drop = [2,3,4,5,6,8,9,13,14] 
        # values_to_drop = [2,3,4,5,6,8,9,13,14,17,21,22,24,33,41,42,
        #                   63,67,78,84,85,86,87,88,90,93,95,96,98,99,101,104,105,
        #                   106,108,113,114,118,120,135,154,158] 
        
        values_to_drop = [344, 345,
                          346,] #cycle not looking good

        # Create a boolean mask to select elements not present in the list
        mask = ~np.isin(unique_45C_cycle, values_to_drop)

        # Filter the array using the mask
        unique_45C_cycle = unique_45C_cycle[mask] 


    for cycle in unique_45C_cycle:

        # cycle = 10
        print(batch_name, ' cycle ',cycle)

        df_median_45C_cycleX = df_median_45C[df_median_45C['cycle']==cycle]
        df_median_45C_cycleX

        # Pick x_data and y_data, ONLY INCLUDE DATA FROM ROW 3 ONWARDS
        x_data = df_median_45C_cycleX['time_h_collapsed'].iloc[2:]
        y_data = df_median_45C_cycleX['PCEmpp_median'].iloc[2:]

        ##### LOGARITHMIC
        # Fit the data to the natural logarithmic equation
        a_fit_log, b_fit_log, c_fit_log = logarithmic_fit_notNorm(x_data,y_data)
        list_a_fit_log.append(a_fit_log)
        list_b_fit_log.append(b_fit_log)
        list_c_fit_log.append(c_fit_log)

        # Generate points for the fitted curve
        y_fit_log = logarithmic_func(x_data, a_fit_log, b_fit_log, c_fit_log)
        r_squared_log =  r_squared(y_data, y_fit_log)
        list_r_squared_log.append(r_squared_log)
        
        ##### LOGARITHMIC +LINEAR
        # Fit the data to the natural logarithmic equation
        a_fit_log_linear, b_fit_log_linear, c_fit_log_linear, d_fit_log_linear = logarithmic_linear_fit_notNorm(x_data,y_data)
        list_a_fit_log_linear.append(a_fit_log_linear)
        list_b_fit_log_linear.append(b_fit_log_linear)
        list_c_fit_log_linear.append(c_fit_log_linear)
        list_d_fit_log_linear.append(d_fit_log_linear)

        # Generate points for the fitted curve
        y_fit_log_linear = logarithmic_linear_func(x_data, a_fit_log_linear, b_fit_log_linear, c_fit_log_linear, d_fit_log_linear)
        r_squared_log_linear =  r_squared(y_data, y_fit_log_linear)
        list_r_squared_log_linear.append(r_squared_log_linear)

        ##### EXP DECAY UPWARDS
        # Fit the data to the exp decay upwards equation
        a_fit_exp, b_fit_exp, c_fit_exp = exp_fit_notNorm(x_data,y_data)
        list_a_fit_exp.append(a_fit_exp)
        list_b_fit_exp.append(b_fit_exp)
        list_c_fit_exp.append(c_fit_exp)

        # Generate points for the fitted curve
        y_fit_exp = exp_func(x_data, a_fit_exp, b_fit_exp, c_fit_exp)
        r_squared_exp =  r_squared(y_data, y_fit_exp)
        list_r_squared_exp.append(r_squared_exp)
        
        ##### EXP DECAY UPWARDS + LINEAR
        # Fit the data to the exp decay upwards equation
        a_fit_exp_linear, b_fit_exp_linear, c_fit_exp_linear, d_fit_exp_linear = exp_linear_fit_notNorm(x_data,y_data)
        list_a_fit_exp_linear.append(a_fit_exp_linear)
        list_b_fit_exp_linear.append(b_fit_exp_linear)
        list_c_fit_exp_linear.append(c_fit_exp_linear)
        list_d_fit_exp_linear.append(d_fit_exp_linear)

        # Generate points for the fitted curve
        y_fit_exp_linear = exp_linear_func(x_data, a_fit_exp_linear, b_fit_exp_linear, c_fit_exp_linear, d_fit_exp_linear)
        r_squared_exp_linear =  r_squared(y_data, y_fit_exp_linear)
        list_r_squared_exp_linear.append(r_squared_exp_linear)

        ##### POWER
        # Fit the data to the power equation
        a_fit_power, b_fit_power = power_fit(x_data,y_data)
        list_a_fit_power.append(a_fit_power)
        list_b_fit_power.append(b_fit_power)

        # Generate points for the fitted curve
        y_fit_power = power_func(x_data, a_fit_power, b_fit_power)
        r_squared_power = r_squared(y_data, y_fit_power)
        list_r_squared_power.append(r_squared_power)

        ##### TANH 
        # Fit the data to the natural logarithmic equation
        a_fit_tanh, b_fit_tanh, c_fit_tanh = tanh_fit(x_data,y_data)
        list_a_fit_tanh.append(a_fit_tanh)
        list_b_fit_tanh.append(b_fit_tanh)
        list_c_fit_tanh.append(c_fit_tanh)

        # Generate points for the fitted curve
        y_fit_tanh = tanh_func(x_data, a_fit_tanh, b_fit_tanh, c_fit_tanh)
        r_squared_tanh =  r_squared(y_data, y_fit_tanh)
        list_r_squared_tanh.append(r_squared_tanh)

        ##### PLOT

        # Create a 3 by 2 subplot and plot the original data and the fitted curve for each function
        plt.close()
        plt.figure(figsize=(15, 8))

        # Subplot 1: Natural Logarithm
        plt.subplot(2, 3, 1)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_log, label='Fitted curve', color='red')
        plt.title(f'Fitted Natural Logarithmic Curve, r2: {r_squared_log:.3f}')
        # plt.legend()
        
        # Subplot 2: Natural Logarithm Linear
        plt.subplot(2, 3, 2)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_log_linear, label='Fitted curve', color='red')
        plt.title(f'Fitted Natural Logarithmic + Linear Curve, r2: {r_squared_log_linear:.3f}')
        # plt.legend()
        
        # Subplot 3: Power Function
        plt.subplot(2, 3, 3)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_power, label='Fitted curve', color='red')
        plt.title(f'Fitted Power Curve, r2: {r_squared_power:.3f}')
        # plt.legend()
        
        # Subplot 4: Exponential Decay Upward
        plt.subplot(2, 3, 4)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_exp, label='Fitted curve', color='red')
        plt.title(f'Fitted Exponential Decay Upwards Curve, r2: {r_squared_exp:.3f}')
        # plt.legend()
        
        # Subplot 5: Exponential Decay Upward Linear
        plt.subplot(2, 3, 5)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_exp_linear, label='Fitted curve', color='red')
        plt.title(f'Fitted Exponential Decay Upwards + Linear Curve, r2: {r_squared_exp_linear:.3f}')
        # plt.legend()

        # Subplot 6: Hyperbolic Tangent (tanh)
        plt.subplot(2, 3, 6)
        sns.scatterplot(x=x_data, y=y_data, label='Original data')
        sns.lineplot(x=x_data, y=y_fit_tanh, label='Fitted curve', color='red')
        plt.title(f'Fitted Tanh Curve, r2: {r_squared_tanh:.3f}')
        plt.legend()
        
        plt.tight_layout()

        # # Ensure that the "figures/curve_fitting" folder exists
        # os.makedirs("figures/20230911_curve_fitting_notNorm", exist_ok=True)

        # Save the figure in the "figures/curve_fitting" folder
        plt.savefig('figures/'+folder_run_name+'cycle_fitting_notNorm/'+batch_name+"_"+str(temperature)+"_cycle_"+str(cycle)+".png",
                    dpi=200)

    #     plt.show()

    # Create the DataFrame
    df_fitting_cycle = pd.DataFrame({
        'cycle':unique_45C_cycle,
        'a_fit_log': list_a_fit_log,
        'b_fit_log': list_b_fit_log,
        'c_fit_log': list_c_fit_log,
        'r_squared_log': list_r_squared_log,
        'a_fit_log_linear': list_a_fit_log_linear,
        'b_fit_log_linear': list_b_fit_log_linear,
        'c_fit_log_linear': list_c_fit_log_linear,
        'd_fit_log_linear': list_d_fit_log_linear,
        'r_squared_log_linear': list_r_squared_log_linear,
        'a_fit_exp': list_a_fit_exp,
        'b_fit_exp': list_b_fit_exp,
        'c_fit_exp': list_c_fit_exp,
        'r_squared_exp': list_r_squared_exp,
        'a_fit_exp_linear': list_a_fit_exp_linear,
        'b_fit_exp_linear': list_b_fit_exp_linear,
        'c_fit_exp_linear': list_c_fit_exp_linear,
        'd_fit_exp_linear': list_d_fit_exp_linear,
        'r_squared_exp_linear': list_r_squared_exp_linear,
        'a_fit_power': list_a_fit_power,
        'b_fit_power': list_b_fit_power,
        'r_squared_power': list_r_squared_power,
        'a_fit_tanh': list_a_fit_tanh,
        'b_fit_tanh': list_b_fit_tanh,
        'c_fit_tanh': list_c_fit_tanh,
        'r_squared_tanh': list_r_squared_tanh,
    })
    

    # Save the dataframe
    df_fitting_cycle.to_csv('figures/'+folder_run_name+'cycle_fitting_notNorm/'+batch_name+"_"+str(temperature)+'_cycle_'+str(cycle)+'.csv',index=False)

    return df_fitting_cycle

def calculate_equivalent_time_using_actual_data(df_fitting_cycle_45, 
                                                df_fitting_cycle_55,
                                                temp_target,
                                                temp_base,
                                                df_calculated_area_45C, 
                                                df_calculated_area_55C,
                                                batch_name,
                                                desired_foldername): # Find equivalent in 45C
    
#     (df_calculated_area_45C['PCEmppArea_median_perHour'])[df_calculated_area_45C['cycle']==1].values[0]
    
    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+desired_foldername):
        os.makedirs('figures/'+folder_run_name+desired_foldername)

    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Find the length of the base
    # if (temp_base == 45) and (temp_target == 55):
    #     desired_length_base = 1.5 # 55C
    # elif (temp_base == 55) and (temp_target == 45):
    #     desired_length_base = 3 # 45C
    # elif temp_base == 35: 
    #     desired_length_base = 3 # 45C
    
    if (temp_target == 55):
        desired_length_base = 1.5 # 55C
    elif (temp_target == 45):
        desired_length_base = 3 # 45C
    elif (temp_target == 35): 
        desired_length_base = 6 # 45C

    list_equivalent_45_time_log = []
    list_equivalent_45_time_log_linear = []
    list_equivalent_45_time_power = []
    list_equivalent_45_time_exp = []
    list_equivalent_45_time_exp_linear = []
    list_equivalent_45_time_tanh = []

    list_equivalent_45_diffArea_log = []
    list_equivalent_45_diffArea_log_linear = []
    list_equivalent_45_diffArea_power = []
    list_equivalent_45_diffArea_exp = []
    list_equivalent_45_diffArea_exp_linear = []
    list_equivalent_45_diffArea_tanh = []
    
    list_equivalent_45_timeLength_target = []
    
    # Create a new list containing elements present in both lists (unique cycles)
    common_cycle = [x for x in unique_45C_cycle if x in unique_55C_cycle]

    # Go through each cycle
    for cycle in unique_45C_cycle:
        
        # If cycle is not on the common cycle:
        if cycle not in common_cycle:
            
            # Save the values
            list_equivalent_45_time_log.append(np.nan)
            list_equivalent_45_diffArea_log.append(np.nan)
            
            list_equivalent_45_time_log_linear.append(np.nan)
            list_equivalent_45_diffArea_log_linear.append(np.nan)
            
            list_equivalent_45_time_power.append(np.nan)
            list_equivalent_45_diffArea_power.append(np.nan)
            
            list_equivalent_45_time_exp.append(np.nan)
            list_equivalent_45_diffArea_exp.append(np.nan)
            
            list_equivalent_45_time_exp_linear.append(np.nan)
            list_equivalent_45_diffArea_exp_linear.append(np.nan)
            
            list_equivalent_45_time_tanh.append(np.nan)
            list_equivalent_45_diffArea_tanh.append(np.nan)
            
            list_equivalent_45_timeLength_target.append(np.nan)
        
        # If cycle is common:
        elif cycle in common_cycle:
            
            ### DESIRED AREA CALCULATION
            desired_area_45 = (df_calculated_area_55C['PCEmppArea_median_perHour'])[df_calculated_area_55C['cycle']==cycle].values[0]
            actual_area_45 = (df_calculated_area_45C['PCEmppArea_median_perHour'])[df_calculated_area_45C['cycle']==cycle].values[0]
            
            # Actual time length of the target
            actual_time_length_45 = (df_calculated_area_45C['length_hour_perCycle'])[df_calculated_area_45C['cycle']==cycle].values[0]
            list_equivalent_45_timeLength_target.append(actual_time_length_45)
            
            ### LOG
            # Parameters of the logarithmic function (replace with your values)
            a_55_log = df_fitting_cycle_55['a_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log = df_fitting_cycle_55['b_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log = df_fitting_cycle_55['c_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log = df_fitting_cycle_45['a_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log = df_fitting_cycle_45['b_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log = df_fitting_cycle_45['c_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log
#             desired_area_45_log = integral_logarithmic_func(desired_length_base, a_55_log, b_55_log, c_55_log)#logarithmic_func(x_55,a_55,b_55,c_55)
            
            # Calculate the value of 'x' log
            result_log = minimize_scalar(area_difference_log, args=(desired_area_45, a_45_log, b_45_log, c_45_log), method='bounded', bounds=(-10.0, 10.0))
            x_value_log = result_log.x

            # Desired area for 55
            area_45_log = integral_logarithmic_func(x_value_log, a_45_log, b_45_log, c_45_log)
            diff_45_log = area_45_log-desired_area_45

            # Save the values
            list_equivalent_45_time_log.append(x_value_log)
            list_equivalent_45_diffArea_log.append(diff_45_log)

        #     print("LOG")
        #     print("LOG: Solution for x from equation:", x_value_log)
        #     print("LOG: Area calculated from the x: ", area_45_log)
        #     print("LOG: The difference: ", diff_45_log)

            ### LOG + LINEAR
            # Parameters of the logarithmic function (replace with your values)
            a_55_log_linear = df_fitting_cycle_55['a_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log_linear = df_fitting_cycle_55['b_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log_linear = df_fitting_cycle_55['c_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_log_linear = df_fitting_cycle_55['d_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log_linear = df_fitting_cycle_45['a_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log_linear = df_fitting_cycle_45['b_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log_linear = df_fitting_cycle_45['c_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_log_linear = df_fitting_cycle_45['d_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log
#             desired_area_45_log = integral_logarithmic_func(desired_length_base, a_55_log, b_55_log, c_55_log)#logarithmic_func(x_55,a_55,b_55,c_55)
            
            # Calculate the value of 'x' log
            result_log_linear = minimize_scalar(area_difference_log_linear, args=(desired_area_45, a_45_log_linear, 
                                                                                  b_45_log_linear, c_45_log_linear,
                                                                                  d_45_log_linear), method='bounded', bounds=(-10.0, 10.0))
            x_value_log_linear = result_log_linear.x

            # Desired area for 55
            area_45_log_linear = integral_logarithmic_linear_func(x_value_log_linear, a_45_log_linear, 
                                                                  b_45_log_linear, c_45_log_linear, d_45_log_linear)
            diff_45_log_linear = area_45_log_linear-desired_area_45

            # Save the values
            list_equivalent_45_time_log_linear.append(x_value_log_linear)
            list_equivalent_45_diffArea_log_linear.append(diff_45_log_linear)


            ### POWER
            # Parameters of the power function (replace with your values)
            a_55_power = df_fitting_cycle_55['a_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_power = df_fitting_cycle_55['b_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_power = df_fitting_cycle_45['a_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_power = df_fitting_cycle_45['b_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 power
#             desired_area_45_power = integral_power_func(desired_length_base, a_55_power, b_55_power)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' power
            result_power = minimize_scalar(area_difference_power, args=(desired_area_45, a_45_power, b_45_power), method='bounded', bounds=(-10.0, 10.0))
            x_value_power = result_power.x

            # Desired area for 55
            area_45_power = integral_power_func(x_value_power, a_45_power, b_45_power)
            diff_45_power = area_45_power-desired_area_45

            # Save the values
            list_equivalent_45_time_power.append(x_value_power)
            list_equivalent_45_diffArea_power.append(diff_45_power)

        #     print("POWER")
        #     print("POWER: Solution for x from equation:", x_value_power)
        #     print("POWER: Area calculated from the x: ", area_45_power)
        #     print("POWER: The difference: ", diff_45_power)

            ### EXP
            # Parameters of the exp function (replace with your values)
            a_55_exp = df_fitting_cycle_55['a_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp = df_fitting_cycle_55['b_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp = df_fitting_cycle_55['c_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp = df_fitting_cycle_45['a_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp = df_fitting_cycle_45['b_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp = df_fitting_cycle_45['c_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
#             desired_area_45_exp = integral_exp_func(desired_length_base, a_55_exp, b_55_exp, c_55_exp)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp = minimize_scalar(area_difference_exp, args=(desired_area_45, a_45_exp, b_45_exp, c_45_exp), method='bounded', bounds=(-10.0, 10.0))
            x_value_exp = result_exp.x

            # Desired area for 55
            area_45_exp = integral_exp_func(x_value_exp, a_45_exp, b_45_exp, c_45_exp)
            diff_45_exp = area_45_exp-desired_area_45

            # Save the values
            list_equivalent_45_time_exp.append(x_value_exp)
            list_equivalent_45_diffArea_exp.append(diff_45_exp)

        #     print("EXP")
        #     print("EXP: Solution for x from equation:", x_value_exp)
        #     print("EXP: Area calculated from the x: ", area_45_exp)
        #     print("EXP: The difference: ", diff_45_exp)

            ### EXP + LINEAR
            # Parameters of the exp function (replace with your values)
            a_55_exp_linear = df_fitting_cycle_55['a_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp_linear = df_fitting_cycle_55['b_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp_linear = df_fitting_cycle_55['c_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_exp_linear = df_fitting_cycle_55['d_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp_linear = df_fitting_cycle_45['a_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp_linear = df_fitting_cycle_45['b_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp_linear = df_fitting_cycle_45['c_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_exp_linear = df_fitting_cycle_45['d_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
#             desired_area_45_exp = integral_exp_func(desired_length_base, a_55_exp, b_55_exp, c_55_exp)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp_linear = minimize_scalar(area_difference_exp_linear, args=(desired_area_45, a_45_exp_linear, 
                                                                                  b_45_exp_linear, c_45_exp_linear,
                                                                                  d_45_exp_linear), method='bounded', bounds=(-10.0, 10.0))
            x_value_exp_linear = result_exp_linear.x

            # Desired area for 55
            area_45_exp_linear = integral_exp_linear_func(x_value_exp_linear, a_45_exp_linear, b_45_exp_linear, 
                                                          c_45_exp_linear, d_45_exp_linear)
            diff_45_exp_linear = area_45_exp_linear-desired_area_45

            # Save the values
            list_equivalent_45_time_exp_linear.append(x_value_exp_linear)
            list_equivalent_45_diffArea_exp_linear.append(diff_45_exp_linear)
            
            ### TANH
            # Parameters of the tanh function (replace with your values)
            a_55_tanh = df_fitting_cycle_55['a_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_tanh = df_fitting_cycle_55['b_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_tanh = df_fitting_cycle_55['c_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_tanh = df_fitting_cycle_45['a_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_tanh = df_fitting_cycle_45['b_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_tanh = df_fitting_cycle_45['c_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 tanh
#             desired_area_45_tanh = integral_tanh_func(desired_length_base, a_55_tanh, b_55_tanh, c_55_tanh)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' tanh
            result_tanh = minimize_scalar(area_difference_tanh, args=(desired_area_45, a_45_tanh, b_45_tanh, c_45_tanh), method='bounded', bounds=(-10.0, 10.0))
            x_value_tanh = result_tanh.x

            # Desired area for 55
            area_45_tanh = integral_exp_func(x_value_tanh, a_45_tanh, b_45_tanh, c_45_tanh)
            diff_45_tanh = area_45_tanh-desired_area_45

            # Save the values
            list_equivalent_45_time_tanh.append(x_value_tanh)
            list_equivalent_45_diffArea_tanh.append(diff_45_tanh)

        #     print("TANH")
        #     print("TANH: Solution for x from equation:", x_value_tanh)
        #     print("TANH: Area calculated from the x: ", area_45_tanh)
        #     print("TANH: The difference: ", diff_45_tanh)
    
    # Prepare for the column name
    str_eq = 'eq_'+str(temp_target)+'C_wrt_'+str(temp_base)+'C_'
    str_eq_time = str_eq + 'time_'
    str_eq_diffArea = str_eq + 'diffArea_'
    list_func = ['log','linear_log','power','exp','linear_exp','tanh']
    
    str_time_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_time]*6, list_func)]
    str_diffArea_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_diffArea]*6, list_func)]
    str_all = str_time_all+ str_diffArea_all

    # Add to the df
    df_fitting_cycle_45[str_all[0]] = list_equivalent_45_time_log
    df_fitting_cycle_45[str_all[1]] = list_equivalent_45_time_log_linear
    df_fitting_cycle_45[str_all[2]] = list_equivalent_45_time_power
    df_fitting_cycle_45[str_all[3]] = list_equivalent_45_time_exp
    df_fitting_cycle_45[str_all[4]] = list_equivalent_45_time_exp_linear
    df_fitting_cycle_45[str_all[5]] = list_equivalent_45_time_tanh

    df_fitting_cycle_45[str_all[4]] = list_equivalent_45_diffArea_log
    df_fitting_cycle_45[str_all[5]] = list_equivalent_45_diffArea_log_linear
    df_fitting_cycle_45[str_all[6]] = list_equivalent_45_diffArea_power
    df_fitting_cycle_45[str_all[7]] = list_equivalent_45_diffArea_exp
    df_fitting_cycle_45[str_all[8]] = list_equivalent_45_diffArea_exp_linear
    df_fitting_cycle_45[str_all[9]] = list_equivalent_45_diffArea_tanh
    
    df_fitting_cycle_45['length_hour_perCycle'] = list_equivalent_45_timeLength_target
    
    # # Multiply the selected columns with the 'actual time length' column and create new columns with '_actual' suffix
    df_fitting_cycle_45 = df_fitting_cycle_45.assign(**{f"{col}_actual": df_fitting_cycle_45[col] * df_fitting_cycle_45['length_hour_perCycle'] for col in str_time_all})
    
    # Save the dataframe
    df_fitting_cycle_45.to_csv('output_dataframe/20230803_fitting_log_notNorm_using_actual_value_'+batch_name+"_eq_Ttarget_"+str(temp_target)+'C_wrt_Tbase_'+str(temp_base)+
                               '_cycle_'+str(cycle)+'.csv',index=False)
    
    
    ### PLOT
    str_eq = str(temp_target)+'C wrt '+str(temp_base)+'C'
    x_data = df_fitting_cycle_45['cycle']
    y_data_log_notActual = df_fitting_cycle_45.filter(like='_time_log').iloc[:,0]
    y_data_log_linear_notActual = df_fitting_cycle_45.filter(like='_time_linear_log').iloc[:,0]
    y_data_power_notActual = df_fitting_cycle_45.filter(like='_time_power').iloc[:,0]
    y_data_exp_notActual = df_fitting_cycle_45.filter(like='_time_exp').iloc[:,0]
    y_data_exp_linear_notActual = df_fitting_cycle_45.filter(like='_time_linear_exp').iloc[:,0]
    y_data_tanh_notActual = df_fitting_cycle_45.filter(like='_time_tanh').iloc[:,0]

    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=y_data_log_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=y_data_log_linear_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    plt.ylim([-5,8])

    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=y_data_power_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 4: Exponential Decay Upward
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=y_data_exp_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 5: Exponential Decay Upward + Linear
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=y_data_exp_linear_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    plt.ylim([-5,8])

    # Subplot 6: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=y_data_tanh_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    plt.ylim([-5,8])

    plt.tight_layout()
    # plt.show()

    # PRINT
    print(str_eq)
    print('Median log: ', y_data_log_notActual.median())
    print('Median log linear: ', y_data_log_linear_notActual.median())
    print('Median exp: ', y_data_exp_notActual.median())
    print('Median exp linear: ', y_data_exp_linear_notActual.median())
    print('Median power: ', y_data_power_notActual.median())
    print('Median tanh: ', y_data_tanh_notActual.median())
    
    
    # # Ensure that the "figures/curve_fitting" folder exists
    # os.makedirs("figures/20230911_curve_fitting", exist_ok=True)

    # Ensure that the "figures/curve_fitting" folder exists
    os.makedirs("figures/20230803_curve_fitting_notNorm", exist_ok=True)

    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig("figures/20230803_curve_fitting_notNorm/fit_equivalent_notNorm_using_actual_value_"+batch_name+"_"+str_eq+".png",
                dpi=200)
    
    plt.close()
    
    # Name txt file
    file_name_txt_notActual = 'figures/'+folder_run_name+desired_foldername+'recap_notNorm_notActual_'+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt_notActual, "w") as file:        

        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log: {y_data_log_notActual.median()} ')
        file.write(f'Median log linear: {y_data_log_linear_notActual.median()} ')
        file.write(f'Median exp: {y_data_exp_notActual.median()} ')
        file.write(f'Median exp linear: {y_data_exp_linear_notActual.median()} ')
        file.write(f'Median power: {y_data_power_notActual.median()} ')
        file.write(f'Median tanh: {y_data_tanh_notActual.median()} ')

    
    ### PLOT ACTUAL
    str_eq = str(temp_target)+'C wrt '+str(temp_base)+'C'
    x_data = df_fitting_cycle_45['cycle']
#     print(df_fitting_cycle_45['length_hour_perCycle'])
    y_data_log = df_fitting_cycle_45.filter(like='_time_log_actual').iloc[:,0]
#     print(y_data_log)
    y_data_log_linear = df_fitting_cycle_45.filter(like='_time_linear_log_actual').iloc[:,0]
    y_data_power = df_fitting_cycle_45.filter(like='_time_power_actual').iloc[:,0]
    y_data_exp = df_fitting_cycle_45.filter(like='_time_exp_actual').iloc[:,0]
    y_data_exp_linear = df_fitting_cycle_45.filter(like='_time_linear_exp_actual').iloc[:,0]
    y_data_tanh = df_fitting_cycle_45.filter(like='_time_tanh_actual').iloc[:,0]

    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=y_data_log, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-0.1,4])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=y_data_log_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    plt.ylim([-0.1,4])
    
    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=y_data_power, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    # Subplot 4: Exponential Decay Upward 
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=y_data_exp, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    # Subplot 5: Exponential Decay Upward 
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=y_data_exp_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    # Subplot 4: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=y_data_tanh, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    plt.tight_layout()
    # plt.show()

    # Name txt file
    file_name_txt = 'figures/'+folder_run_name+desired_foldername+'recap_fit_equivalent_notNorm_actualValue_'+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt, "w") as file:        
    
        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log: {y_data_log.median()} ')
        file.write(f'Median log linear: {y_data_log_linear.median()} ')
        file.write(f'Median exp: {y_data_exp.median()} ')
        file.write(f'Median exp_linear: {y_data_exp_linear.median()} ')
        file.write(f'Median power: {y_data_power.median()} ')
        file.write(f'Median tanh: {y_data_tanh.median()} ')
    
    
    # # Ensure that the "figures/curve_fitting" folder exists
    # os.makedirs("figures/20230911_curve_fitting", exist_ok=True)
    
    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig('figures/'+folder_run_name+desired_foldername+
                'fit_equivalent_notNorm_actualValue_'+
                batch_name+"_"+str_eq+".png",
                dpi=200)
    
    return df_fitting_cycle_45


# notNorm

def calculate_equivalent_time_using_actual_data_notNorm(df_fitting_cycle_45, 
                                                        df_fitting_cycle_55,
                                                        temp_target,
                                                        temp_base,
                                                        df_calculated_area_45C, 
                                                        df_calculated_area_55C,
                                                        batch_name,
                                                        desired_foldername): # Find equivalent in 45C
    
#     (df_calculated_area_45C['PCEmppArea_median_perHour'])[df_calculated_area_45C['cycle']==1].values[0]
    
    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+desired_foldername):
        os.makedirs('figures/'+folder_run_name+desired_foldername)

    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Find the length of the base
    # if (temp_base == 45) and (temp_target == 55):
    #     desired_length_base = 1.5 # 55C
    # elif (temp_base == 55) and (temp_target == 45):
    #     desired_length_base = 3 # 45C
    # elif temp_base == 35: 
    #     desired_length_base = 3 # 45C
    
    if (temp_target == 55):
        desired_length_base = 1.5 # 55C
    elif (temp_target == 45):
        desired_length_base = 3 # 45C
    elif (temp_target == 35): 
        desired_length_base = 6 # 45C
    elif (temp_target == 25): 
        desired_length_base = 12 # 45C

    list_equivalent_45_time_log = []
    list_equivalent_45_time_log_linear = []
    list_equivalent_45_time_power = []
    list_equivalent_45_time_exp = []
    list_equivalent_45_time_exp_linear = []
    list_equivalent_45_time_tanh = []

    list_equivalent_45_diffArea_log = []
    list_equivalent_45_diffArea_log_linear = []
    list_equivalent_45_diffArea_power = []
    list_equivalent_45_diffArea_exp = []
    list_equivalent_45_diffArea_exp_linear = []
    list_equivalent_45_diffArea_tanh = []
    
    list_equivalent_45_timeLength_target = []
    
    # Create a new list containing elements present in both lists (unique cycles)
    common_cycle = [x for x in unique_45C_cycle if x in unique_55C_cycle]

    # Go through each cycle
    for cycle in unique_45C_cycle:
        
        # If cycle is not on the common cycle:
        if cycle not in common_cycle:
            
            # Save the values
            list_equivalent_45_time_log.append(np.nan)
            list_equivalent_45_diffArea_log.append(np.nan)
            
            list_equivalent_45_time_log_linear.append(np.nan)
            list_equivalent_45_diffArea_log_linear.append(np.nan)
            
            list_equivalent_45_time_power.append(np.nan)
            list_equivalent_45_diffArea_power.append(np.nan)
            
            list_equivalent_45_time_exp.append(np.nan)
            list_equivalent_45_diffArea_exp.append(np.nan)
            
            list_equivalent_45_time_exp_linear.append(np.nan)
            list_equivalent_45_diffArea_exp_linear.append(np.nan)
            
            list_equivalent_45_time_tanh.append(np.nan)
            list_equivalent_45_diffArea_tanh.append(np.nan)
            
            list_equivalent_45_timeLength_target.append(np.nan)
        
        # If cycle is common:
        elif cycle in common_cycle:
            
            ### DESIRED AREA CALCULATION
            
            # These ones are actual value
            desired_area_45 = (df_calculated_area_55C['PCEmppArea_median_perHour'])[df_calculated_area_55C['cycle']==cycle].values[0]
            actual_area_45 = (df_calculated_area_45C['PCEmppArea_median_perHour'])[df_calculated_area_45C['cycle']==cycle].values[0]
            
            # Actual time length of the target
            actual_time_length_45 = (df_calculated_area_45C['length_hour_perCycle'])[df_calculated_area_45C['cycle']==cycle].values[0]
            list_equivalent_45_timeLength_target.append(actual_time_length_45)
            
            ### LOG
            # Parameters of the logarithmic function (replace with your values)
            a_55_log = df_fitting_cycle_55['a_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log = df_fitting_cycle_55['b_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log = df_fitting_cycle_55['c_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log = df_fitting_cycle_45['a_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log = df_fitting_cycle_45['b_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log = df_fitting_cycle_45['c_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log/ based on the function
            desired_area_45_log = integral_logarithmic_func(desired_length_base, a_55_log, b_55_log, c_55_log)#logarithmic_func(x_55,a_55,b_55,c_55)
            
            # Calculate the value of 'x' log
            result_log = minimize_scalar(area_difference_log, args=(desired_area_45_log, a_45_log, b_45_log, c_45_log), method='bounded', bounds=(-10.0, 10.0))
            x_value_log = result_log.x

            # Desired area for 55
            area_45_log = integral_logarithmic_func(x_value_log, a_45_log, b_45_log, c_45_log)
            diff_45_log = area_45_log-desired_area_45

            # Save the values
            list_equivalent_45_time_log.append(x_value_log)
            list_equivalent_45_diffArea_log.append(diff_45_log)

            ### LOG + LINEAR
            # Parameters of the logarithmic function (replace with your values)
            a_55_log_linear = df_fitting_cycle_55['a_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log_linear = df_fitting_cycle_55['b_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log_linear = df_fitting_cycle_55['c_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_log_linear = df_fitting_cycle_55['d_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log_linear = df_fitting_cycle_45['a_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log_linear = df_fitting_cycle_45['b_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log_linear = df_fitting_cycle_45['c_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_log_linear = df_fitting_cycle_45['d_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log
#             desired_area_45_log = integral_logarithmic_func(desired_length_base, a_55_log, b_55_log, c_55_log)#logarithmic_func(x_55,a_55,b_55,c_55)
            
            # Calculate the value of 'x' log
            result_log_linear = minimize_scalar(area_difference_log_linear, args=(desired_area_45, a_45_log_linear, 
                                                                                  b_45_log_linear, c_45_log_linear,
                                                                                  d_45_log_linear), method='bounded', bounds=(-10.0, 10.0))
            x_value_log_linear = result_log_linear.x

            # Desired area for 55
            area_45_log_linear = integral_logarithmic_linear_func(x_value_log_linear, a_45_log_linear, 
                                                                  b_45_log_linear, c_45_log_linear, d_45_log_linear)
            diff_45_log_linear = area_45_log_linear-desired_area_45

            # Save the values
            list_equivalent_45_time_log_linear.append(x_value_log_linear)
            list_equivalent_45_diffArea_log_linear.append(diff_45_log_linear)


            ### POWER
            # Parameters of the power function (replace with your values)
            a_55_power = df_fitting_cycle_55['a_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_power = df_fitting_cycle_55['b_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_power = df_fitting_cycle_45['a_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_power = df_fitting_cycle_45['b_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 power
#             desired_area_45_power = integral_power_func(desired_length_base, a_55_power, b_55_power)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' power
            result_power = minimize_scalar(area_difference_power, args=(desired_area_45, a_45_power, b_45_power), method='bounded', bounds=(-10.0, 10.0))
            x_value_power = result_power.x

            # Desired area for 55
            area_45_power = integral_power_func(x_value_power, a_45_power, b_45_power)
            diff_45_power = area_45_power-desired_area_45

            # Save the values
            list_equivalent_45_time_power.append(x_value_power)
            list_equivalent_45_diffArea_power.append(diff_45_power)

            ### EXP
            # Parameters of the exp function (replace with your values)
            a_55_exp = df_fitting_cycle_55['a_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp = df_fitting_cycle_55['b_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp = df_fitting_cycle_55['c_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp = df_fitting_cycle_45['a_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp = df_fitting_cycle_45['b_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp = df_fitting_cycle_45['c_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
#             desired_area_45_exp = integral_exp_func(desired_length_base, a_55_exp, b_55_exp, c_55_exp)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp = minimize_scalar(area_difference_exp, args=(desired_area_45, a_45_exp, b_45_exp, c_45_exp), method='bounded', bounds=(-10.0, 10.0))
            x_value_exp = result_exp.x

            # Desired area for 55
            area_45_exp = integral_exp_func(x_value_exp, a_45_exp, b_45_exp, c_45_exp)
            diff_45_exp = area_45_exp-desired_area_45

            # Save the values
            list_equivalent_45_time_exp.append(x_value_exp)
            list_equivalent_45_diffArea_exp.append(diff_45_exp)


            ### EXP + LINEAR
            # Parameters of the exp function (replace with your values)
            a_55_exp_linear = df_fitting_cycle_55['a_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp_linear = df_fitting_cycle_55['b_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp_linear = df_fitting_cycle_55['c_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_exp_linear = df_fitting_cycle_55['d_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp_linear = df_fitting_cycle_45['a_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp_linear = df_fitting_cycle_45['b_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp_linear = df_fitting_cycle_45['c_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_exp_linear = df_fitting_cycle_45['d_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
#             desired_area_45_exp = integral_exp_func(desired_length_base, a_55_exp, b_55_exp, c_55_exp)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp_linear = minimize_scalar(area_difference_exp_linear, args=(desired_area_45, a_45_exp_linear, 
                                                                                  b_45_exp_linear, c_45_exp_linear,
                                                                                  d_45_exp_linear), method='bounded', bounds=(-10.0, 10.0))
            x_value_exp_linear = result_exp_linear.x

            # Desired area for 55
            area_45_exp_linear = integral_exp_linear_func(x_value_exp_linear, a_45_exp_linear, b_45_exp_linear, 
                                                          c_45_exp_linear, d_45_exp_linear)
            diff_45_exp_linear = area_45_exp_linear-desired_area_45

            # Save the values
            list_equivalent_45_time_exp_linear.append(x_value_exp_linear)
            list_equivalent_45_diffArea_exp_linear.append(diff_45_exp_linear)
            
            ### TANH
            # Parameters of the tanh function (replace with your values)
            a_55_tanh = df_fitting_cycle_55['a_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_tanh = df_fitting_cycle_55['b_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_tanh = df_fitting_cycle_55['c_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_tanh = df_fitting_cycle_45['a_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_tanh = df_fitting_cycle_45['b_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_tanh = df_fitting_cycle_45['c_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 tanh
#             desired_area_45_tanh = integral_tanh_func(desired_length_base, a_55_tanh, b_55_tanh, c_55_tanh)#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' tanh
            result_tanh = minimize_scalar(area_difference_tanh, args=(desired_area_45, a_45_tanh, b_45_tanh, c_45_tanh), method='bounded', bounds=(-10.0, 10.0))
            x_value_tanh = result_tanh.x

            # Desired area for 55
            area_45_tanh = integral_exp_func(x_value_tanh, a_45_tanh, b_45_tanh, c_45_tanh)
            diff_45_tanh = area_45_tanh-desired_area_45

            # Save the values
            list_equivalent_45_time_tanh.append(x_value_tanh)
            list_equivalent_45_diffArea_tanh.append(diff_45_tanh)
    
    # Prepare for the column name
    str_eq = 'eq_'+str(temp_target)+'C_wrt_'+str(temp_base)+'C_'
    str_eq_time = str_eq + 'time_'
    str_eq_diffArea = str_eq + 'diffArea_'
    list_func = ['log','linear_log','power','exp','linear_exp','tanh']
    
    str_time_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_time]*6, list_func)]
    str_diffArea_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_diffArea]*6, list_func)]
    str_all = str_time_all+ str_diffArea_all

    # Add to the df
    df_fitting_cycle_45[str_all[0]] = list_equivalent_45_time_log
    df_fitting_cycle_45[str_all[1]] = list_equivalent_45_time_log_linear
    df_fitting_cycle_45[str_all[2]] = list_equivalent_45_time_power
    df_fitting_cycle_45[str_all[3]] = list_equivalent_45_time_exp
    df_fitting_cycle_45[str_all[4]] = list_equivalent_45_time_exp_linear
    df_fitting_cycle_45[str_all[5]] = list_equivalent_45_time_tanh

    df_fitting_cycle_45[str_all[4]] = list_equivalent_45_diffArea_log
    df_fitting_cycle_45[str_all[5]] = list_equivalent_45_diffArea_log_linear
    df_fitting_cycle_45[str_all[6]] = list_equivalent_45_diffArea_power
    df_fitting_cycle_45[str_all[7]] = list_equivalent_45_diffArea_exp
    df_fitting_cycle_45[str_all[8]] = list_equivalent_45_diffArea_exp_linear
    df_fitting_cycle_45[str_all[9]] = list_equivalent_45_diffArea_tanh
    
    df_fitting_cycle_45['length_hour_perCycle'] = list_equivalent_45_timeLength_target
    
    # # Multiply the selected columns with the 'actual time length' column and create new columns with '_actual' suffix
    df_fitting_cycle_45 = df_fitting_cycle_45.assign(**{f"{col}_actual": df_fitting_cycle_45[col] * df_fitting_cycle_45['length_hour_perCycle'] for col in str_time_all})
    
    # Save the dataframe
    df_fitting_cycle_45.to_csv('output_dataframe/20230803_fitting_log_notNorm_using_actual_value_'+batch_name+"_eq_Ttarget_"+str(temp_target)+'C_wrt_Tbase_'+str(temp_base)+
                               '_cycle_'+str(cycle)+'.csv',index=False)
    
    
    ### PLOT
    str_eq = str(temp_target)+'C wrt '+str(temp_base)+'C'
    x_data = df_fitting_cycle_45['cycle']
    y_data_log_notActual = df_fitting_cycle_45.filter(like='_time_log').iloc[:,0]
    y_data_log_linear_notActual = df_fitting_cycle_45.filter(like='_time_linear_log').iloc[:,0]
    y_data_power_notActual = df_fitting_cycle_45.filter(like='_time_power').iloc[:,0]
    y_data_exp_notActual = df_fitting_cycle_45.filter(like='_time_exp').iloc[:,0]
    y_data_exp_linear_notActual = df_fitting_cycle_45.filter(like='_time_linear_exp').iloc[:,0]
    y_data_tanh_notActual = df_fitting_cycle_45.filter(like='_time_tanh').iloc[:,0]

    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=y_data_log_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=y_data_log_linear_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    plt.ylim([-5,8])

    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=y_data_power_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 4: Exponential Decay Upward
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=y_data_exp_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 5: Exponential Decay Upward + Linear
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=y_data_exp_linear_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    plt.ylim([-5,8])

    # Subplot 6: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=y_data_tanh_notActual, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    plt.ylim([-5,8])

    plt.tight_layout()
    # plt.show()

    # PRINT
    print(str_eq)
    print('Median log: ', y_data_log_notActual.median())
    print('Median log linear: ', y_data_log_linear_notActual.median())
    print('Median exp: ', y_data_exp_notActual.median())
    print('Median exp linear: ', y_data_exp_linear_notActual.median())
    print('Median power: ', y_data_power_notActual.median())
    print('Median tanh: ', y_data_tanh_notActual.median())
    
    
    # # Ensure that the "figures/curve_fitting" folder exists
    # os.makedirs("figures/20230911_curve_fitting", exist_ok=True)

    # Ensure that the "figures/curve_fitting" folder exists
    os.makedirs("figures/20230803_curve_fitting_notNorm", exist_ok=True)

    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig("figures/20230803_curve_fitting_notNorm/fit_equivalent_notNorm_using_actual_value_"+batch_name+"_"+str_eq+".png",
                dpi=200)
    
    plt.close()
    
    # Name txt file
    file_name_txt_notActual = 'figures/'+folder_run_name+desired_foldername+'recap_notNorm_notActual_'+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt_notActual, "w") as file:        

        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log: {y_data_log_notActual.median()} ')
        file.write(f'Median log linear: {y_data_log_linear_notActual.median()} ')
        file.write(f'Median exp: {y_data_exp_notActual.median()} ')
        file.write(f'Median exp linear: {y_data_exp_linear_notActual.median()} ')
        file.write(f'Median power: {y_data_power_notActual.median()} ')
        file.write(f'Median tanh: {y_data_tanh_notActual.median()} ')

    
    ### PLOT ACTUAL
    str_eq = str(temp_target)+'C wrt '+str(temp_base)+'C'
    x_data = df_fitting_cycle_45['cycle']
#     print(df_fitting_cycle_45['length_hour_perCycle'])
    y_data_log = df_fitting_cycle_45.filter(like='_time_log_actual').iloc[:,0]
#     print(y_data_log)
    y_data_log_linear = df_fitting_cycle_45.filter(like='_time_linear_log_actual').iloc[:,0]
    y_data_power = df_fitting_cycle_45.filter(like='_time_power_actual').iloc[:,0]
    y_data_exp = df_fitting_cycle_45.filter(like='_time_exp_actual').iloc[:,0]
    y_data_exp_linear = df_fitting_cycle_45.filter(like='_time_linear_exp_actual').iloc[:,0]
    y_data_tanh = df_fitting_cycle_45.filter(like='_time_tanh_actual').iloc[:,0]

    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=y_data_log, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-0.1,4])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=y_data_log_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    plt.ylim([-0.1,4])
    
    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=y_data_power, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    # Subplot 4: Exponential Decay Upward 
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=y_data_exp, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    # Subplot 5: Exponential Decay Upward 
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=y_data_exp_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    # Subplot 4: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=y_data_tanh, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    plt.ylim([-0.1,4])

    plt.tight_layout()
    # plt.show()

    # Name txt file
    file_name_txt = 'figures/'+folder_run_name+desired_foldername+'recap_fit_equivalent_notNorm_actualValue_'+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt, "w") as file:        
    
        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log: {y_data_log.median()} ')
        file.write(f'Median log linear: {y_data_log_linear.median()} ')
        file.write(f'Median exp: {y_data_exp.median()} ')
        file.write(f'Median exp_linear: {y_data_exp_linear.median()} ')
        file.write(f'Median power: {y_data_power.median()} ')
        file.write(f'Median tanh: {y_data_tanh.median()} ')
    
    
    # # Ensure that the "figures/curve_fitting" folder exists
    # os.makedirs("figures/20230911_curve_fitting", exist_ok=True)
    
    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig('figures/'+folder_run_name+desired_foldername+
                'fit_equivalent_notNorm_actualValue_'+
                batch_name+"_"+str_eq+".png",
                dpi=200)
    
    return df_fitting_cycle_45

# Fitting for the not norm aging cycle curves 
def fit_aging_cycle_function_notNorm(df_all_25C, df_all_35C, df_all_45C, df_all_55C):
    
    batchname_list_tracking = []
    
    # Go through each batch name
    for batch_name in batchname_list:
        
        # Select specific batch for each temperature
        df_batch_selected_25C = df_all_25C[(df_all_25C['temperature']==25) & (df_all_25C['device_name']==batch_name)]
        df_batch_selected_35C = df_all_35C[(df_all_35C['temperature']==35) & (df_all_35C['device_name']==batch_name)]
        df_batch_selected_45C = df_all_45C[(df_all_45C['temperature']==45) & (df_all_45C['device_name']==batch_name)]
        df_batch_selected_55C = df_all_55C[(df_all_55C['temperature']==55) & (df_all_55C['device_name']==batch_name)]

        # Go through each row
        df_median_55C = ((df_batch_selected_55C['MPPT'].iloc[0])['time_h']).to_frame()
        df_median_45C = ((df_batch_selected_45C['MPPT'].iloc[0])['time_h']).to_frame()
        df_median_35C = ((df_batch_selected_35C['MPPT'].iloc[0])['time_h']).to_frame()
        df_median_25C = ((df_batch_selected_25C['MPPT'].iloc[0])['time_h']).to_frame()
        
        df_median_55C['time_h_collapsed']=((df_batch_selected_55C['MPPT'].iloc[0])['time_h_collapsed'])
        df_median_45C['time_h_collapsed']=((df_batch_selected_45C['MPPT'].iloc[0])['time_h_collapsed'])
        df_median_35C['time_h_collapsed']=((df_batch_selected_35C['MPPT'].iloc[0])['time_h_collapsed'])
        df_median_25C['time_h_collapsed']=((df_batch_selected_25C['MPPT'].iloc[0])['time_h_collapsed'])
        
        df_median_55C['cycle']=((df_batch_selected_55C['MPPT'].iloc[0])['cycle'])
        df_median_45C['cycle']=((df_batch_selected_45C['MPPT'].iloc[0])['cycle'])
        df_median_35C['cycle']=((df_batch_selected_35C['MPPT'].iloc[0])['cycle'])
        df_median_25C['cycle']=((df_batch_selected_25C['MPPT'].iloc[0])['cycle'])

        df_median_55C = calculate_median_specific_T(df_batch_selected_55C, df_median_55C)
        df_median_45C = calculate_median_specific_T(df_batch_selected_45C, df_median_45C)
        df_median_35C = calculate_median_specific_T(df_batch_selected_35C, df_median_35C)
        df_median_25C = calculate_median_specific_T(df_batch_selected_25C, df_median_25C)
        
        # Need to divide by # max of the column? Yes
        max_value_25C = df_median_25C['PCEmpp_median'].max()
        max_value_35C = df_median_35C['PCEmpp_median'].max()
        max_value_45C = df_median_45C['PCEmpp_median'].max()
        max_value_55C = df_median_55C['PCEmpp_median'].max()
        
        df_median_25C['PCEmpp_median_norm'] = df_median_25C['PCEmpp_median']/max_value_25C
        df_median_35C['PCEmpp_median_norm'] = df_median_35C['PCEmpp_median']/max_value_35C
        df_median_45C['PCEmpp_median_norm'] = df_median_45C['PCEmpp_median']/max_value_45C
        df_median_55C['PCEmpp_median_norm'] = df_median_55C['PCEmpp_median']/max_value_55C
                
        # Fit all cycles
        df_fitting_cycle_25C = fit_all_cycles_notNorm(df_median_25C, 25, batch_name)
        df_fitting_cycle_35C = fit_all_cycles_notNorm(df_median_35C, 35, batch_name)
        df_fitting_cycle_45C = fit_all_cycles_notNorm(df_median_45C, 45, batch_name)
        df_fitting_cycle_55C = fit_all_cycles_notNorm(df_median_55C, 55, batch_name)
        
        batchname_list_tracking.append([batch_name,
                                        df_fitting_cycle_25C,
                                        df_fitting_cycle_35C,
                                        df_fitting_cycle_45C,
                                        df_fitting_cycle_55C,
                                        df_median_25C,
                                        df_median_35C,
                                        df_median_45C, 
                                        df_median_55C])
        
    return batchname_list_tracking


###############################################################################
# FUNCTIONS: BACK-CALCULATE FOR EQUIVALENT OF T, ACTUAL VALUE/ CUMSUM
###############################################################################

# Function to calculcate cumsum
def calculate_cumsum_median(df_median):
    
    whole_selected_df = pd.DataFrame()
    
    # Go through each cycle
    cycle_unique = df_median['cycle'].unique()

    for cycle in cycle_unique:
        selected_df = df_median[df_median['cycle']==cycle]

        # New list
        list_trapz_PCEmpp = []
        list_trapz_PCEmpp_norm = []
#         list_trapz_Vmpp = []

        for index,row in selected_df.iterrows():

            if index == selected_df.index[0]:
                # Since there's no value yet for the integration
                list_trapz_PCEmpp.append(0)
                list_trapz_PCEmpp_norm.append(0)
#                 list_trapz_Vmpp.append(0)
            else:
                trapz_val_PCEmpp = 0.5*(selected_df['time_h'][index]-selected_df['time_h'][index-1])*(selected_df['PCEmpp_median'][index]+selected_df['PCEmpp_median'][index-1])
                trapz_val_PCEmpp_norm = 0.5*(selected_df['time_h'][index]-selected_df['time_h'][index-1])*(selected_df['PCEmpp_median_norm'][index]+selected_df['PCEmpp_median_norm'][index-1])
#                 trapz_val_Vmpp = 0.5*(selected_df['time_h'][index]-selected_df['time_h'][index-1])*(selected_df['V_mpp'][index]+selected_df['V_mpp'][index-1])

                list_trapz_PCEmpp.append(trapz_val_PCEmpp)
                list_trapz_PCEmpp_norm.append(trapz_val_PCEmpp_norm)

        # Add list to the dataframe
        selected_df['trapz_PCEmpp_median'] = list_trapz_PCEmpp
        selected_df['trapz_PCEmpp_median_norm'] = list_trapz_PCEmpp_norm
#         selected_df['trapz_Vmpp'] = list_trapz_Vmpp
        selected_df['trapz_PCEmpp_median_perHour'] = list_trapz_PCEmpp/selected_df['time_h_collapsed'].max()
        selected_df['trapz_PCEmpp_median_norm_perHour'] = list_trapz_PCEmpp_norm/selected_df['time_h_collapsed'].max()
#         selected_df['trapz_Vmpp_perHour'] = list_trapz_Vmpp/selected_df['time_h_collapsed'].max()

        # Cumsum
        selected_df['cumsum_PCEmpp_median'] = selected_df['trapz_PCEmpp_median'].cumsum()
        selected_df['cumsum_PCEmpp_median_norm'] = selected_df['trapz_PCEmpp_median_norm'].cumsum()
#         selected_df['cumsum_Vmpp'] = selected_df['trapz_Vmpp'].cumsum()
        selected_df['cumsum_PCEmpp_median_perHour'] = selected_df['trapz_PCEmpp_median_perHour'].cumsum()
        selected_df['cumsum_PCEmpp_median_norm_perHour'] = selected_df['trapz_PCEmpp_median_norm_perHour'].cumsum()
#         selected_df['cumsum_Vmpp_perHour'] = selected_df['trapz_Vmpp_perHour'].cumsum()

#         # Cumsum normalized
#         selected_df['cumsum_PCEmpp_perHour_norm'] = selected_df['cumsum_PCEmpp_perHour']/df_all_25C_45C_55C_area['PCEmppArea_perCycle_perHour_max'][indexdf]
#         selected_df['cumsum_Vmpp_perHour_norm'] = selected_df['cumsum_Vmpp_perHour']/df_all_25C_45C_55C_area['VmppArea_perCycle_perHour_max'][indexdf]

        # Append dataframe
        whole_selected_df = pd.concat([whole_selected_df, selected_df])
    
    return whole_selected_df


# Function to calculate max of the cumsum
def cumsum_median_max(cumsum_median):
    
    # New dataframe
    df_max = pd.DataFrame()
    
    # Go through each cycle
    cycle_unique = cumsum_median['cycle'].unique()
    
    # List for max value
    max_value_list = []
    
    for cycle in cycle_unique:
        
        selected_df = cumsum_median[cumsum_median['cycle']==cycle]
        max_value = selected_df['cumsum_PCEmpp_median'].max()
        max_value_list.append(max_value)
    
    df_max['cycle'] = cycle_unique
    df_max['cumsum_PCEmpp_median_max'] = max_value_list
    
    return df_max
    
# Function to find equivalent area    
def find_cumsum_median_in_45C(cumsum_median_45C, cumsum_median_55C_max):
    
    # cumsum_median_45C: baseline, 55C: target
    
    # Find overlapping unique cycle
    cycle_unique_45C = cumsum_median_45C['cycle'].unique()
    cycle_unique_55C = cumsum_median_55C_max['cycle'].unique()
    
    # Convert lists to set
    set_cycle_45C = set(cycle_unique_45C)
    set_cycle_55C = set(cycle_unique_55C)
    
    # Find overlapping elements
    overlap_cycle = set_cycle_45C.intersection(set_cycle_55C)
    
    # List cycle
    equivalent_45C_value = []
    equivalent_45C_time_h = []
    equivalent_45C_time_h_collapsed = []
    actual_55C_value = []
    
    # Go through the overlap cycle
    for cycle in overlap_cycle:
        
        # Selected 45C df
        selected_45C_df = cumsum_median_45C[cumsum_median_45C['cycle']==cycle]
        
        # Value of interest
        value_max_55C = cumsum_median_55C_max[cumsum_median_55C_max['cycle'] == cycle]['cumsum_PCEmpp_median_max'].values[0]
        actual_55C_value.append(value_max_55C)
        
        # Tolerance
#         value_max_55C_tol = value_max_55C*tolerance
        
        # Pick the row        
        matching_rows = selected_45C_df[(selected_45C_df['cumsum_PCEmpp_median'] >= value_max_55C)]
        
        if not matching_rows.empty: 
            time_h_matching = matching_rows['time_h'].iloc[0]
            time_h_collapsed_matching = matching_rows['time_h_collapsed'].iloc[0]
            cumsum_PCEmpp_median_matching = matching_rows['cumsum_PCEmpp_median'].iloc[0]
            
            equivalent_45C_time_h.append(time_h_matching)
            equivalent_45C_time_h_collapsed.append(time_h_collapsed_matching)
            equivalent_45C_value.append(cumsum_PCEmpp_median_matching)
            
        else:
            equivalent_45C_time_h.append(np.nan)
            equivalent_45C_time_h_collapsed.append(np.nan)
            equivalent_45C_value.append(np.nan)
      
    df_analytical = pd.DataFrame({
        'cycle': list(overlap_cycle),
        'time_h_equivalent': equivalent_45C_time_h,
        'time_h_collapsed_equivalent': equivalent_45C_time_h_collapsed,
        'cumsum_PCEmpp_median_equivalent': equivalent_45C_value,
        'cumsum_PCEmpp_median_actual': actual_55C_value,
    })
    
    df_analytical['delta_cumsum_PCEmpp_median'] = df_analytical['cumsum_PCEmpp_median_equivalent'] - df_analytical['cumsum_PCEmpp_median_actual']
    
    return df_analytical


# Function to plot equivalent time across cycles wrt specific T
def plot_equivalent_cumsum(df_analytical_wrt, T_baseline, T_target, batch_name, desired_foldername):
    
    plt.close()
    plt.figure(figsize=(5,3.5))
    
    sns.scatterplot(x=df_analytical_wrt['cycle'],
                    y=df_analytical_wrt['time_h_collapsed_equivalent'],
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Equivalent {T_target}C wrt {T_baseline}C {batch_name}')
    # plt.ylim([-0.1,4])
    
    # Saving figure
    plt.savefig('figures/'+folder_run_name+desired_foldername+
                batch_name+'_equivalent_'+str(T_target)+'C_wrt_'+str(T_baseline)+
                'C_analytical.png', dpi=600)



###############################################################################
# FUNCTIONS: RUNNING SUBSEQUENT FUNCTIONS
###############################################################################

# Run loading all data for all temperatures
def run_load_data_allTs(path_25C, path_45C, path_55C, cell_params_35C, limit_h_35C):
    
    # Load data from the folders into df
    df_all_25C = load_data_into_df(hour_cycle_25C, gap_25C, temperature_25C, path_25C)
    df_all_45C = load_data_into_df(hour_cycle_45C, gap_45C, temperature_45C, path_45C)
    df_all_55C = load_data_into_df(hour_cycle_55C, gap_55C, temperature_55C, path_55C)
    df_all_35C = load_data_from_python_into_df(hour_cycle_35C, gap_35C, temperature_35C, 
                                               path_35C, cell_params_35C, limit_h=limit_h_35C)
    
    return df_all_25C, df_all_35C, df_all_45C, df_all_55C

def run_load_data_python_allTs(path_25C, path_35C, path_45C, path_55C, cell_params_25C,
                               cell_params_35C, cell_params_45C, cell_params_55C, 
                               limit_h_25C, limit_h_35C, limit_h_45C, limit_h_55C):
    
    # Load data from the folders into df WITH PYTHON
    # df_all_25C = load_data_into_df(hour_cycle_25C, gap_25C, temperature_25C, path_25C)
    # df_all_45C = load_data_into_df(hour_cycle_45C, gap_45C, temperature_45C, path_45C)
    # df_all_55C = load_data_into_df(hour_cycle_55C, gap_55C, temperature_55C, path_55C)
    
    df_all_25C = load_data_from_python_into_df(hour_cycle_25C, gap_25C, temperature_25C, 
                                               path_25C, cell_params_25C, limit_h=limit_h_25C)
    df_all_35C = load_data_from_python_into_df(hour_cycle_35C, gap_35C, temperature_35C, 
                                               path_35C, cell_params_35C, limit_h=limit_h_35C)
    
    df_all_45C = load_data_from_python_into_df(hour_cycle_45C, gap_45C, temperature_45C, 
                                               path_45C, cell_params_45C, limit_h=limit_h_45C)
    df_all_55C = load_data_from_python_into_df(hour_cycle_55C, gap_55C, temperature_55C, 
                                               path_55C, cell_params_55C, limit_h=limit_h_55C)
    
    return df_all_25C, df_all_35C, df_all_45C, df_all_55C


# Run calculating area for each cycle for all temperatures
def run_calculate_area_allTs(df_all_25C, df_all_35C, df_all_45C, df_all_55C):
    
    # Calculate for all the df_all
    df_all_25C_area = calculate_area_perCycle(df_all_25C)
    df_all_35C_area = calculate_area_perCycle(df_all_35C)
    df_all_45C_area = calculate_area_perCycle(df_all_45C)
    df_all_55C_area = calculate_area_perCycle(df_all_55C)
    
    # Concatenate everything
    df_all_25C_35C_45C_55C_area = (pd.concat([df_all_25C_area, df_all_35C_area,
                                              df_all_45C_area, df_all_55C_area])).reset_index(drop=True)
        
    return(df_all_25C_area, df_all_35C_area, 
           df_all_45C_area, df_all_55C_area, df_all_25C_35C_45C_55C_area)


# Run calculating statistics for all the areas 
def run_calculate_area_stats_allTs(df_all_25C_area, df_all_35C_area, 
                                   df_all_45C_area, df_all_55C_area):
    
    df_stats_25C = calculate_area_perCycle_stats(df_all_25C_area)
    df_stats_35C = calculate_area_perCycle_stats(df_all_35C_area)
    df_stats_45C = calculate_area_perCycle_stats(df_all_45C_area)
    df_stats_55C = calculate_area_perCycle_stats(df_all_55C_area)
    
    # Concatenate everything
    df_all_stats = (pd.concat([df_stats_25C, df_stats_35C,
                               df_stats_45C, df_stats_55C])).reset_index(drop=True)

    
    return(df_stats_25C, df_stats_35C, df_stats_45C, df_stats_55C, df_all_stats)
    

# Run calculating comparison between Vmpp and PCEmpp 0 vs 1000h
def run_calculate_0vs1000h_allTs(df_all_25C, df_all_35C, df_all_45C, df_all_55C):

    df_0_1000_25C = calculate_Vmpp_PCEmpp_0_1000(df_all_25C)
    df_0_1000_35C = calculate_Vmpp_PCEmpp_0_1000(df_all_35C)
    df_0_1000_45C = calculate_Vmpp_PCEmpp_0_1000(df_all_45C)
    df_0_1000_55C = calculate_Vmpp_PCEmpp_0_1000(df_all_55C)
    
    return (df_0_1000_25C, df_0_1000_35C, df_0_1000_45C, df_0_1000_55C)


# Run the back calculate (NUMERICAL) based on the fitted equations
def run_calculate_equivalent_time_all_batches(batch_fitting, desired_foldername): 
    
    list_all_backcalculations = []
    
    # Go through all the lists
    for i in range(len(batchname_list)):
        
        batch_name = batch_fitting[i][0]
        df_fitting_cycle_35 = batch_fitting[i][1]
        df_fitting_cycle_45 = batch_fitting[i][2]
        df_fitting_cycle_55 = batch_fitting[i][3]
        
        
        df_backcalculate_target_45_base_35 = calculate_equivalent_time(df_fitting_cycle_45,
                                                                       df_fitting_cycle_35,
                                                                       45,
                                                                       35,
                                                                       batch_name,
                                                                       desired_foldername)
        
        df_backcalculate_target_55_base_35 = calculate_equivalent_time(df_fitting_cycle_55,
                                                                       df_fitting_cycle_35,
                                                                       55,
                                                                       35,
                                                                       batch_name,
                                                                       desired_foldername)
        
        df_backcalculate_target_35_base_45 = calculate_equivalent_time(df_fitting_cycle_35,
                                                                       df_fitting_cycle_45,
                                                                       35,
                                                                       45,
                                                                       batch_name,
                                                                       desired_foldername)
        
        df_backcalculate_target_55_base_45 = calculate_equivalent_time(df_fitting_cycle_55,
                                                                       df_fitting_cycle_45,
                                                                       55,
                                                                       45,
                                                                       batch_name,
                                                                       desired_foldername)
        
        df_backcalculate_target_35_base_55 = calculate_equivalent_time(df_fitting_cycle_35,
                                                                       df_fitting_cycle_55,
                                                                       35,
                                                                       55,
                                                                       batch_name,
                                                                       desired_foldername)
        
        df_backcalculate_target_45_base_55 = calculate_equivalent_time(df_fitting_cycle_45,
                                                                       df_fitting_cycle_55,
                                                                       45,
                                                                       55,
                                                                       batch_name)
        
        # Create dictionary for all the results
        df_backcalculation = {
            'batch_name':batch_name,
            'backcalculate_target_45_base_35': df_backcalculate_target_45_base_35,
            'backcalculate_target_55_base_35': df_backcalculate_target_55_base_35,
            'backcalculate_target_35_base_45': df_backcalculate_target_35_base_45,
            'backcalculate_target_55_base_45': df_backcalculate_target_55_base_45,
            'backcalculate_target_35_base_55': df_backcalculate_target_35_base_55,
            'backcalculate_target_45_base_55': df_backcalculate_target_45_base_55,
            }
        
        # Append to list
        list_all_backcalculations.append(df_backcalculation)
    
    return(list_all_backcalculations)


def run_calculate_equivalent_time_all_batches_notNorm(batch_fitting_notNorm, desired_foldername): 
    
    list_all_backcalculations = []
    
    # Go through all the lists
    for i in range(len(batchname_list)):
        
        batch_name = batch_fitting_notNorm[i][0]
        df_fitting_cycle_35 = batch_fitting_notNorm[i][1]
        df_fitting_cycle_45 = batch_fitting_notNorm[i][2]
        df_fitting_cycle_55 = batch_fitting_notNorm[i][3]
        
        df_median_35C = batch_fitting_notNorm[i][4]
        df_median_45C = batch_fitting_notNorm[i][5]
        df_median_55C = batch_fitting_notNorm[i][6]
        
        df_calculated_area_35C = create_df_area_under_PCEmpp_median(df_median_35C)
        df_calculated_area_45C = create_df_area_under_PCEmpp_median(df_median_45C)
        df_calculated_area_55C = create_df_area_under_PCEmpp_median(df_median_55C)
        
        
        df_backcalculate_target_45_base_35 = calculate_equivalent_time_using_actual_data(df_fitting_cycle_45,
                                                                                         df_fitting_cycle_35,
                                                                                         45,
                                                                                         35,
                                                                                         df_calculated_area_45C, 
                                                                                         df_calculated_area_35C,
                                                                                         batch_name,
                                                                                         desired_foldername)
        
        df_backcalculate_target_55_base_35 = calculate_equivalent_time_using_actual_data(df_fitting_cycle_55,
                                                                                         df_fitting_cycle_35,
                                                                                         55,
                                                                                         35,
                                                                                         df_calculated_area_55C, 
                                                                                         df_calculated_area_35C,
                                                                                         batch_name,
                                                                                         desired_foldername)
        
        df_backcalculate_target_35_base_45 = calculate_equivalent_time_using_actual_data(df_fitting_cycle_35,
                                                                                         df_fitting_cycle_45,
                                                                                         35,
                                                                                         45,
                                                                                         df_calculated_area_35C, 
                                                                                         df_calculated_area_45C,
                                                                                         batch_name,
                                                                                         desired_foldername)
        
        df_backcalculate_target_55_base_45 = calculate_equivalent_time_using_actual_data(df_fitting_cycle_55,
                                                                                         df_fitting_cycle_45,
                                                                                         55,
                                                                                         45,
                                                                                         df_calculated_area_55C, 
                                                                                         df_calculated_area_45C,
                                                                                         batch_name,
                                                                                         desired_foldername)
        
        df_backcalculate_target_35_base_55 = calculate_equivalent_time_using_actual_data(df_fitting_cycle_35,
                                                                                         df_fitting_cycle_55,
                                                                                         35,
                                                                                         55,
                                                                                         df_calculated_area_35C, 
                                                                                         df_calculated_area_55C,
                                                                                         batch_name,
                                                                                         desired_foldername)
        
        df_backcalculate_target_45_base_55 = calculate_equivalent_time_using_actual_data(df_fitting_cycle_45,
                                                                                         df_fitting_cycle_55,
                                                                                         45,
                                                                                         55,
                                                                                         df_calculated_area_45C, 
                                                                                         df_calculated_area_55C,
                                                                                         batch_name,
                                                                                         desired_foldername)
        
        # Create dictionary for all the results
        df_backcalculation = {
            'batch_name':batch_name,
            'backcalculate_target_45_base_35': df_backcalculate_target_45_base_35,
            'backcalculate_target_55_base_35': df_backcalculate_target_55_base_35,
            'backcalculate_target_35_base_45': df_backcalculate_target_35_base_45,
            'backcalculate_target_55_base_45': df_backcalculate_target_55_base_45,
            'backcalculate_target_35_base_55': df_backcalculate_target_35_base_55,
            'backcalculate_target_45_base_55': df_backcalculate_target_45_base_55,
            }
        
        # Append to list
        list_all_backcalculations.append(df_backcalculation)
    
    return(list_all_backcalculations)

def calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_45, 
                                                        df_fitting_cycle_55,
                                                        temp_target,
                                                        temp_base,
                                                        df_calculated_area_45C, 
                                                        df_calculated_area_55C,
                                                        batch_name,
                                                        desired_foldername): # Find equivalent in 45C
    
    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+desired_foldername):
        os.makedirs('figures/'+folder_run_name+desired_foldername)

    unique_45C_cycle = df_fitting_cycle_45['cycle'].unique()
    unique_55C_cycle = df_fitting_cycle_55['cycle'].unique()
    
    # Find the length of the base
    
    if (temp_target == 55):
        desired_length_target = 1.5 # 55C
    elif (temp_target == 45):
        desired_length_target = 3 # 45C
    elif (temp_target == 35): 
        desired_length_target = 6 # 35C
    elif (temp_target == 25): 
        desired_length_target = 12 # 25C
        
    
    if (temp_base == 55):
        desired_length_base = 1.5 # 55C
        exclude_index = 118
    elif (temp_base == 45):
        desired_length_base = 3 # 45C
        exclude_index = 59
    elif (temp_base == 35): 
        desired_length_base = 6 # 35C
        exclude_index=9
    elif (temp_base == 25): 
        desired_length_base = 12 # 25C
        exclude_index = 14

    list_equivalent_45_time_log = []
    list_equivalent_45_time_log_linear = []
    list_equivalent_45_time_power = []
    list_equivalent_45_time_exp = []
    list_equivalent_45_time_exp_linear = []
    list_equivalent_45_time_tanh = []

    list_equivalent_45_diffArea_log = []
    list_equivalent_45_diffArea_log_linear = []
    list_equivalent_45_diffArea_power = []
    list_equivalent_45_diffArea_exp = []
    list_equivalent_45_diffArea_exp_linear = []
    list_equivalent_45_diffArea_tanh = []
    
    list_equivalent_45_timeLength_target = []
    
    # Create a new list containing elements present in both lists (unique cycles)
    common_cycle = [x for x in unique_45C_cycle if x in unique_55C_cycle]

    # Go through each cycle
    for cycle in unique_45C_cycle:
        
        # If cycle is not on the common cycle:
        if cycle not in common_cycle:
            
            # Save the values
            list_equivalent_45_time_log.append(np.nan)
            list_equivalent_45_diffArea_log.append(np.nan)
            
            list_equivalent_45_time_log_linear.append(np.nan)
            list_equivalent_45_diffArea_log_linear.append(np.nan)
            
            list_equivalent_45_time_power.append(np.nan)
            list_equivalent_45_diffArea_power.append(np.nan)
            
            list_equivalent_45_time_exp.append(np.nan)
            list_equivalent_45_diffArea_exp.append(np.nan)
            
            list_equivalent_45_time_exp_linear.append(np.nan)
            list_equivalent_45_diffArea_exp_linear.append(np.nan)
            
            list_equivalent_45_time_tanh.append(np.nan)
            list_equivalent_45_diffArea_tanh.append(np.nan)
            
            list_equivalent_45_timeLength_target.append(np.nan)
        
        # If cycle is common:
        elif cycle in common_cycle:
            
            ### DESIRED AREA CALCULATION
            
            # These ones are actual value
            desired_area_45 = (df_calculated_area_55C['PCEmppArea_median_perHour'])[df_calculated_area_55C['cycle']==cycle].values[0]
            actual_area_45 = (df_calculated_area_45C['PCEmppArea_median_perHour'])[df_calculated_area_45C['cycle']==cycle].values[0]
            
            # Actual time length of the target
            actual_time_length_45 = (df_calculated_area_45C['length_hour_perCycle'])[df_calculated_area_45C['cycle']==cycle].values[0]
            list_equivalent_45_timeLength_target.append(actual_time_length_45)
            
            ### LOG
            # Parameters of the logarithmic function (replace with your values)
            a_55_log = df_fitting_cycle_55['a_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log = df_fitting_cycle_55['b_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log = df_fitting_cycle_55['c_fit_log'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log = df_fitting_cycle_45['a_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log = df_fitting_cycle_45['b_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log = df_fitting_cycle_45['c_fit_log'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log/ based on the function
            desired_area_45_log = integral_logarithmic_func(desired_length_base, a_55_log, b_55_log, c_55_log)/desired_length_base#logarithmic_func(x_55,a_55,b_55,c_55)
            
            # Calculate the value of 'x' log
            result_log = minimize_scalar(area_difference_log_notNorm, args=(desired_area_45_log, a_45_log, b_45_log, c_45_log))#, method='bounded', bounds=(-10.0, 10.0))
            x_value_log = result_log.x
            # print('x_value_log: ',x_value_log)

            # Desired area for 55
            area_45_log = integral_logarithmic_func(x_value_log, a_45_log, b_45_log, c_45_log)
            diff_45_log = area_45_log/x_value_log-desired_area_45_log #diff: normalized

            # Save the values
            list_equivalent_45_time_log.append(x_value_log)
            list_equivalent_45_diffArea_log.append(diff_45_log)

        #     print("LOG")
        #     print("LOG: Solution for x from equation:", x_value_log)
        #     print("LOG: Area calculated from the x: ", area_45_log)
        #     print("LOG: The difference: ", diff_45_log)

            ### LOG + LINEAR
            # Parameters of the logarithmic function (replace with your values)
            a_55_log_linear = df_fitting_cycle_55['a_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_log_linear = df_fitting_cycle_55['b_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_log_linear = df_fitting_cycle_55['c_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_log_linear = df_fitting_cycle_55['d_fit_log_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_log_linear = df_fitting_cycle_45['a_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_log_linear = df_fitting_cycle_45['b_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_log_linear = df_fitting_cycle_45['c_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_log_linear = df_fitting_cycle_45['d_fit_log_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 log
            desired_area_45_log_linear = integral_logarithmic_linear_func(desired_length_base, a_55_log_linear, b_55_log_linear,
                                                                          c_55_log_linear,d_55_log_linear)/desired_length_base#logarithmic_func(x_55,a_55,b_55,c_55)
            
            # Calculate the value of 'x' log
            result_log_linear = minimize_scalar(area_difference_log_linear_notNorm, args=(desired_area_45_log_linear, a_45_log_linear,
                                                                                          b_45_log_linear, c_45_log_linear,
                                                                                          d_45_log_linear))#, method='bounded', bounds=(-10.0, 10.0))
            x_value_log_linear = result_log_linear.x
            # print('x_value_log_linear: ',x_value_log_linear)

            # Desired area for 55
            area_45_log_linear = integral_logarithmic_linear_func(x_value_log_linear, a_45_log_linear, 
                                                                  b_45_log_linear, c_45_log_linear, d_45_log_linear)
            diff_45_log_linear = area_45_log_linear/x_value_log_linear-desired_area_45_log_linear

            # Save the values
            list_equivalent_45_time_log_linear.append(x_value_log_linear)
            list_equivalent_45_diffArea_log_linear.append(diff_45_log_linear)


            ### POWER
            # Parameters of the power function (replace with your values)
            a_55_power = df_fitting_cycle_55['a_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_power = df_fitting_cycle_55['b_fit_power'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_power = df_fitting_cycle_45['a_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_power = df_fitting_cycle_45['b_fit_power'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 power
            desired_area_45_power = integral_power_func(desired_length_base, a_55_power, b_55_power)/desired_length_base#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' power
            result_power = minimize_scalar(area_difference_power_notNorm, args=(desired_area_45_power, a_45_power, b_45_power))#, method='bounded', bounds=(-10.0, 10.0))
            x_value_power = result_power.x
            # print('x_value_power: ', x_value_power)

            # Desired area for 55
            area_45_power = integral_power_func(x_value_power, a_45_power, b_45_power)
            diff_45_power = area_45_power/x_value_power-desired_area_45

            # Save the values
            list_equivalent_45_time_power.append(x_value_power)
            list_equivalent_45_diffArea_power.append(diff_45_power)

        #     print("POWER")
        #     print("POWER: Solution for x from equation:", x_value_power)
        #     print("POWER: Area calculated from the x: ", area_45_power)
        #     print("POWER: The difference: ", diff_45_power)

            ### EXP
            # Parameters of the exp function (replace with your values)
            a_55_exp = df_fitting_cycle_55['a_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp = df_fitting_cycle_55['b_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp = df_fitting_cycle_55['c_fit_exp'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp = df_fitting_cycle_45['a_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp = df_fitting_cycle_45['b_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp = df_fitting_cycle_45['c_fit_exp'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
            desired_area_45_exp = integral_exp_func(desired_length_base, a_55_exp, b_55_exp, c_55_exp)/desired_length_base#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp = minimize_scalar(area_difference_exp_notNorm, args=(desired_area_45_exp, a_45_exp, b_45_exp, c_45_exp))#, method='bounded', bounds=(-10.0, 10.0))
            x_value_exp = result_exp.x
            # print('x_value_exp:',x_value_exp)

            # Desired area for 55
            area_45_exp = integral_exp_func(x_value_exp, a_45_exp, b_45_exp, c_45_exp)
            diff_45_exp = area_45_exp/x_value_exp-desired_area_45

            # Save the values
            list_equivalent_45_time_exp.append(x_value_exp)
            list_equivalent_45_diffArea_exp.append(diff_45_exp)

        #     print("EXP")
        #     print("EXP: Solution for x from equation:", x_value_exp)
        #     print("EXP: Area calculated from the x: ", area_45_exp)
        #     print("EXP: The difference: ", diff_45_exp)

            ### EXP + LINEAR
            # Parameters of the exp function (replace with your values)
            a_55_exp_linear = df_fitting_cycle_55['a_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_exp_linear = df_fitting_cycle_55['b_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_exp_linear = df_fitting_cycle_55['c_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]
            d_55_exp_linear = df_fitting_cycle_55['d_fit_exp_linear'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_exp_linear = df_fitting_cycle_45['a_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_exp_linear = df_fitting_cycle_45['b_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_exp_linear = df_fitting_cycle_45['c_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]
            d_45_exp_linear = df_fitting_cycle_45['d_fit_exp_linear'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 exp
            desired_area_45_exp_linear = integral_exp_linear_func(desired_length_base, a_55_exp_linear, b_55_exp_linear, c_55_exp_linear,
                                                                  d_55_exp_linear)/desired_length_base#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' exp
            result_exp_linear = minimize_scalar(area_difference_exp_linear_notNorm, args=(desired_area_45_exp_linear, a_45_exp_linear, 
                                                                                          b_45_exp_linear, c_45_exp_linear,
                                                                                          d_45_exp_linear))#, method='bounded', bounds=(-10.0, 10.0))
            x_value_exp_linear = result_exp_linear.x
            # print('x_value_exp_linear: ',x_value_exp_linear)

            # Desired area for 55
            area_45_exp_linear = integral_exp_linear_func(x_value_exp_linear, a_45_exp_linear, b_45_exp_linear, 
                                                          c_45_exp_linear, d_45_exp_linear)
            diff_45_exp_linear = area_45_exp_linear/x_value_exp_linear-desired_area_45

            # Save the values
            list_equivalent_45_time_exp_linear.append(x_value_exp_linear)
            list_equivalent_45_diffArea_exp_linear.append(diff_45_exp_linear)
            
            ### TANH
            # Parameters of the tanh function (replace with your values)
            a_55_tanh = df_fitting_cycle_55['a_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            b_55_tanh = df_fitting_cycle_55['b_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]
            c_55_tanh = df_fitting_cycle_55['c_fit_tanh'][df_fitting_cycle_55['cycle']==cycle].values[0]

            a_45_tanh = df_fitting_cycle_45['a_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            b_45_tanh = df_fitting_cycle_45['b_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]
            c_45_tanh = df_fitting_cycle_45['c_fit_tanh'][df_fitting_cycle_45['cycle']==cycle].values[0]

            # Desired area for 45 tanh
            desired_area_45_tanh = integral_tanh_func(desired_length_base, a_55_tanh, b_55_tanh, c_55_tanh)/desired_length_base#logarithmic_func(x_55,a_55,b_55,c_55)

            # Calculate the value of 'x' tanh
            result_tanh = minimize_scalar(area_difference_tanh_notNorm, args=(desired_area_45_tanh, a_45_tanh, b_45_tanh, c_45_tanh))#, method='bounded', bounds=(-10.0, 10.0))
            x_value_tanh = result_tanh.x
            # print('x_value_tanh: ',x_value_tanh)

            # Desired area for 55
            area_45_tanh = integral_exp_func(x_value_tanh, a_45_tanh, b_45_tanh, c_45_tanh)
            diff_45_tanh = area_45_tanh/x_value_tanh-desired_area_45

            # Save the values
            list_equivalent_45_time_tanh.append(x_value_tanh)
            list_equivalent_45_diffArea_tanh.append(diff_45_tanh)

        #     print("TANH")
        #     print("TANH: Solution for x from equation:", x_value_tanh)
        #     print("TANH: Area calculated from the x: ", area_45_tanh)
        #     print("TANH: The difference: ", diff_45_tanh)
    
    # Prepare for the column name
    str_eq = 'eq_'+str(temp_target)+'C_wrt_'+str(temp_base)+'C_'
    str_eq_time = str_eq + 'time_'
    str_eq_diffArea = str_eq + 'diffArea_'
    list_func = ['log','linear_log','power','exp','linear_exp','tanh']
    
    str_time_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_time]*6, list_func)]
    
    str_diffArea_all = [elem1 + elem2 for elem1, elem2 in zip([str_eq_diffArea]*6, list_func)]
    str_all = str_time_all+ str_diffArea_all
    # print(str_all)

    # Create a dictionary with the str_all names and list of the values
    df_time_equivalent = {
        str_all[0]: list_equivalent_45_time_log,
        str_all[1]: list_equivalent_45_time_log_linear,
        str_all[2]: list_equivalent_45_time_power,
        str_all[3]: list_equivalent_45_time_exp,
        str_all[4]: list_equivalent_45_time_exp_linear,
        str_all[5]: list_equivalent_45_time_tanh,
        str_all[6]: list_equivalent_45_diffArea_log,
        str_all[7]: list_equivalent_45_diffArea_log_linear,
        str_all[8]: list_equivalent_45_diffArea_power,
        str_all[9]: list_equivalent_45_diffArea_exp,
        str_all[10]: list_equivalent_45_diffArea_exp_linear,
        str_all[11]: list_equivalent_45_diffArea_tanh,
        }
    
    # Create a DataFrame from the dictionary
    df_time_equivalent = pd.DataFrame(df_time_equivalent)
    
    # Combine the 2 dfs: df_fitting_cycle_45, and df_time_equivalent
    combined_df_fitting_time_equivalent = pd.concat([df_fitting_cycle_45, 
                                                     df_time_equivalent], axis=1)
    
    # Save the dataframe
    combined_df_fitting_time_equivalent.to_csv('output_dataframe/20230803_fitting_log_notNorm_using_actual_value_'+batch_name+"_eq_Ttarget_"+
                                               str(temp_target)+'C_wrt_Tbase_'+str(temp_base)+'_cycle_'+str(cycle)+'.csv',index=False)
    
    
    ### PLOT
    str_eq = str(temp_target)+'C wrt '+str(temp_base)+'C'
    x_data = df_fitting_cycle_45['cycle']
    y_data_log_notActual = df_time_equivalent.filter(like='_time_log').iloc[:,0]
    print(f"shape of y_data_log_notActual: {y_data_log_notActual.shape} ")
    # Check the shape oh each y_data_log
    y_data_log_linear_notActual = df_time_equivalent.filter(like='_time_linear_log').iloc[:,0]
    print(f"shape of y_data_log_linear_notActual: {y_data_log_linear_notActual.shape} ")
    
    y_data_power_notActual = df_time_equivalent.filter(like='_time_power').iloc[:,0]
    print(f"shape of y_data_log_linear_notActual: {y_data_power_notActual.shape} ")
    
    y_data_exp_notActual = df_time_equivalent.filter(like='_time_exp').iloc[:,0]
    print(f"shape of y_data_log_linear_notActual: {y_data_exp_notActual.shape} ")
    
    y_data_exp_linear_notActual = df_time_equivalent.filter(like='_time_linear_exp').iloc[:,0]
    print(f"shape of y_data_log_linear_notActual: {y_data_exp_linear_notActual.shape} ")
    
    y_data_tanh_notActual = df_time_equivalent.filter(like='_time_tanh').iloc[:,0]
    print(f"shape of y_data_log_linear_notActual: {y_data_tanh_notActual.shape} ")

    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_log, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_log_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    # plt.ylim([-5,8])

    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_power, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    # plt.ylim([-5,8])
    
    # Subplot 4: Exponential Decay Upward
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_exp, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    # plt.ylim([-5,8])
    
    # Subplot 5: Exponential Decay Upward + Linear
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_exp_linear, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    # plt.ylim([-5,8])

    # Subplot 6: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_tanh, label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    # plt.ylim([-5,8])

    plt.tight_layout()
    # plt.show()
    
    # # Ensure that the "figures/curve_fitting" folder exists
    # os.makedirs("figures/20230911_curve_fitting", exist_ok=True)

    # Ensure that the "figures/curve_fitting" folder exists
    os.makedirs("figures/"+folder_run_name+"backcalculate_notNorm/", exist_ok=True)
    os.makedirs("figures/"+folder_run_name+"backcalculate_notNorm_exc35C/", exist_ok=True)

    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig("figures/"+folder_run_name+desired_foldername+"fit_equivalent_notNorm_"+batch_name+"_"+str_eq+".png",
                dpi=600)
    
    plt.close()
    
    # PRINT
    print(str_eq)
    print('Median log: ', y_data_log_notActual.median())
    print(f'25th percentile log: {np.percentile(y_data_log_notActual, 25)} ')
    print(f'75th percentile log: {np.percentile(y_data_log_notActual, 75)} ')
    
    print('Median log linear: ', y_data_log_linear_notActual.median())
    print(f'25th percentile log linear: {np.percentile(y_data_log_linear_notActual, 25)} ')
    print(f'75th percentile log linear: {np.percentile(y_data_log_linear_notActual, 75)} ')
    
    print('Median exp: ', y_data_exp_notActual.median())
    print(f'25th percentile exp: {np.percentile(y_data_exp_notActual, 25)} ')
    print(f'75th percentile exp: {np.percentile(y_data_exp_notActual, 75)} ')
    
    print('Median exp linear: ', y_data_exp_linear_notActual.median())
    print(f'25th percentile exp linear: {np.percentile(y_data_exp_linear_notActual, 25)} ')
    print(f'75th percentile exp linear: {np.percentile(y_data_exp_linear_notActual, 75)} ')
    
    print('Median power: ', y_data_power_notActual.median())
    print(f'25th percentile power: {np.percentile(y_data_power_notActual, 25)} ')
    print(f'75th percentile power: {np.percentile(y_data_power_notActual, 75)} ')    
    
    print('Median tanh: ', y_data_tanh_notActual.median())
    print(f'25th percentile tanh: {np.percentile(y_data_tanh_notActual, 25)} ')
    print(f'75th percentile tanh: {np.percentile(y_data_tanh_notActual, 75)} ')    
    
    
    # Name txt file
    file_name_txt_notActual = 'figures/'+folder_run_name+desired_foldername+'recap_notNorm_notActual_'+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt_notActual, "w") as file:        

        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log: {y_data_log_notActual.median()} ')
        file.write(f'25th percentile log: {np.percentile(y_data_log_notActual, 25)} ')
        file.write(f'75th percentile log: {np.percentile(y_data_log_notActual, 75)} ')
        
        file.write(f'Median log linear: {y_data_log_linear_notActual.median()} ')
        file.write(f'25th percentile log linear: {np.percentile(y_data_log_linear_notActual, 25)} ')
        file.write(f'75th percentile log linear: {np.percentile(y_data_log_linear_notActual, 75)} ')
        
        file.write(f'Median exp: {y_data_exp_notActual.median()} ')
        file.write(f'25th percentile exp: {np.percentile(y_data_exp_notActual, 25)} ')
        file.write(f'75th percentile exp: {np.percentile(y_data_exp_notActual, 75)} ')
        
        file.write(f'Median exp linear: {y_data_exp_linear_notActual.median()} ')
        file.write(f'25th percentile exp linear: {np.percentile(y_data_exp_linear_notActual, 25)} ')
        file.write(f'75th percentile exp linear: {np.percentile(y_data_exp_linear_notActual, 75)} ')
        
        file.write(f'Median power: {y_data_power_notActual.median()} ')
        file.write(f'25th percentile power: {np.percentile(y_data_power_notActual, 25)} ')
        file.write(f'75th percentile power: {np.percentile(y_data_power_notActual, 75)} ')
        
        file.write(f'Median tanh: {y_data_tanh_notActual.median()} ')
        file.write(f'25th percentile tanh: {np.percentile(y_data_tanh_notActual, 25)} ')
        file.write(f'75th percentile tanh: {np.percentile(y_data_tanh_notActual, 75)} ')
    
    # Create a 2 by 2 subplot and plot the original data and the fitted curve for each function
    plt.close()
    plt.figure(figsize=(15, 8))

    # Subplot 1: Natural Logarithm
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_log,
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic fit equivalent {str_eq}')
    plt.ylim([-5,8])
    
    # Subplot 2: Natural Logarithm + Linear
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_log_linear,
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Logarithmic linear fit equivalent {str_eq}')
    # plt.ylim([-5,8])

    # Subplot 3: Power Function
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_power,
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Power fit equivalent {str_eq}')
    # plt.ylim([-5,8])
    
    # Subplot 4: Exponential Decay Upward
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_exp,
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward fit equivalent {str_eq}')
    # plt.ylim([-5,8])
    
    # Subplot 5: Exponential Decay Upward + Linear
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_exp_linear,
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Exponential decay upward linear fit equivalent {str_eq}')
    # plt.ylim([-5,8])

    # Subplot 6: Hyperbolic Tangent (tanh)
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=x_data, y=list_equivalent_45_time_tanh,
                    label='Equivalent time',alpha=0.6,edgecolor = None)
    plt.title(f'Tanh fit equivalent {str_eq}')
    # plt.ylim([-5,8])

    plt.tight_layout()
    # plt.show()
    
    # # Ensure that the "figures/curve_fitting" folder exists
    # os.makedirs("figures/20230911_curve_fitting", exist_ok=True)

    # Ensure that the "figures/curve_fitting" folder exists
    os.makedirs("figures/"+folder_run_name+"backcalculate_notNorm/", exist_ok=True)

    # Save the figure in the "figures/curve_fitting" folder
    plt.savefig("figures/"+folder_run_name+"backcalculate_notNorm/"+
                "fit_equivalent_notNorm_incFirstThird_"+batch_name+"_"+str_eq+".png",
                dpi=600)   
    
    
    
    # Name txt file
    file_name_txt_notActual = 'figures/'+folder_run_name+"backcalculate_notNorm/recap_notNorm_inc_FirstThird_notActual_"+batch_name+'_'+str_eq+'.txt'
    
    # Store the results
    with open(file_name_txt_notActual, "w") as file:        

        file.write(f"{batch_name}, target:{temp_target}C, base: {temp_base}C ")
        file.write(f'Median log subset: {np.median(list_equivalent_45_time_log)} ')
        file.write(f'25th percentile log subset: {np.percentile(list_equivalent_45_time_log, 25)} ')
        file.write(f'75th percentile log subset: {np.percentile(list_equivalent_45_time_log, 75)} ')
        
        file.write(f'Median log linear subset: {np.median(list_equivalent_45_time_log_linear)} ')
        file.write(f'25th percentile log linear subset: {np.percentile(list_equivalent_45_time_log_linear, 25)} ')
        file.write(f'75th percentile log linear subset: {np.percentile(list_equivalent_45_time_log_linear, 75)} ')
        
        file.write(f'Median exp subset: {np.median(list_equivalent_45_time_power)} ')
        file.write(f'25th percentile exp subset: {np.percentile(list_equivalent_45_time_power, 25)} ')
        file.write(f'75th percentile exp subset: {np.percentile(list_equivalent_45_time_power, 75)} ')
        
        file.write(f'Median exp linear subset: {np.median(list_equivalent_45_time_exp)} ')
        file.write(f'25th percentile exp linear subset: {np.percentile(list_equivalent_45_time_exp, 25)} ')
        file.write(f'75th percentile exp linear subset: {np.percentile(list_equivalent_45_time_exp, 75)} ')
        
        file.write(f'Median power subset: {np.median(list_equivalent_45_time_exp_linear)} ')
        file.write(f'25th percentile power subset: {np.percentile(list_equivalent_45_time_exp_linear, 25)} ')
        file.write(f'75th percentile power subset: {np.percentile(list_equivalent_45_time_exp_linear, 75)} ')
        
        file.write(f'Median tanh subset: {np.median(list_equivalent_45_time_tanh)} ')
        file.write(f'25th percentile tanh subset: {np.percentile(list_equivalent_45_time_tanh, 25)} ')
        file.write(f'75th percentile tanh subset: {np.percentile(list_equivalent_45_time_tanh, 75)} ')
    
    return y_data_log_linear_notActual,df_fitting_cycle_45

def run_calculate_equivalent_time_all_batches_notNorm_v2(batch_fitting_notNorm, desired_foldername): 
    '''
    Version 2: based on 2023/10/27
    Not norm based on the functions (desired area using functions)
    '''
    
    list_all_backcalculations = []
    
    # Go through all the lists
    for i in range(len(batchname_list)):
        
        batch_name = batch_fitting_notNorm[i][0]
        df_fitting_cycle_25 = batch_fitting_notNorm[i][1]
        df_fitting_cycle_35 = batch_fitting_notNorm[i][2]
        df_fitting_cycle_45 = batch_fitting_notNorm[i][3]
        df_fitting_cycle_55 = batch_fitting_notNorm[i][4]
        
        df_median_25C = batch_fitting_notNorm[i][5]
        df_median_35C = batch_fitting_notNorm[i][6]
        df_median_45C = batch_fitting_notNorm[i][7]
        df_median_55C = batch_fitting_notNorm[i][8]
        
        df_calculated_area_25C = create_df_area_under_PCEmpp_median(df_median_25C)
        df_calculated_area_35C = create_df_area_under_PCEmpp_median(df_median_35C)
        df_calculated_area_45C = create_df_area_under_PCEmpp_median(df_median_45C)
        df_calculated_area_55C = create_df_area_under_PCEmpp_median(df_median_55C)
        
        # Base 25C
        df_backcalculate_target_35_base_25 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_35,
                                                                                                  df_fitting_cycle_25,
                                                                                                  35,
                                                                                                  25,
                                                                                                  df_calculated_area_35C, 
                                                                                                  df_calculated_area_25C,
                                                                                                  batch_name,
                                                                                                  desired_foldername)
        
        df_backcalculate_target_45_base_25 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_45,
                                                                                                  df_fitting_cycle_25,
                                                                                                  45,
                                                                                                  25,
                                                                                                  df_calculated_area_45C, 
                                                                                                  df_calculated_area_25C,
                                                                                                  batch_name,
                                                                                                  desired_foldername)
        
        df_backcalculate_target_55_base_25 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_55,
                                                                                                  df_fitting_cycle_25,
                                                                                                  55,
                                                                                                  25,
                                                                                                  df_calculated_area_55C, 
                                                                                                  df_calculated_area_25C,
                                                                                                  batch_name,
                                                                                                  desired_foldername)
        
        # Base 35C        
        df_backcalculate_target_25_base_35 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_25,
                                                                                                  df_fitting_cycle_35,
                                                                                                  25,
                                                                                                  35,
                                                                                                  df_calculated_area_25C, 
                                                                                                  df_calculated_area_35C,
                                                                                                  batch_name,
                                                                                                  desired_foldername)
        
        df_backcalculate_target_45_base_35 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_45,
                                                                                                 df_fitting_cycle_35,
                                                                                                 45,
                                                                                                 35,
                                                                                                 df_calculated_area_45C, 
                                                                                                 df_calculated_area_35C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        df_backcalculate_target_55_base_35 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_55,
                                                                                                 df_fitting_cycle_35,
                                                                                                 55,
                                                                                                 35,
                                                                                                 df_calculated_area_55C, 
                                                                                                 df_calculated_area_35C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        # Base 45C
        df_backcalculate_target_25_base_45 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_25,
                                                                                                 df_fitting_cycle_45,
                                                                                                 25,
                                                                                                 45,
                                                                                                 df_calculated_area_25C, 
                                                                                                 df_calculated_area_45C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        
        df_backcalculate_target_35_base_45 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_35,
                                                                                                 df_fitting_cycle_45,
                                                                                                 35,
                                                                                                 45,
                                                                                                 df_calculated_area_35C, 
                                                                                                 df_calculated_area_45C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        df_backcalculate_target_55_base_45 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_55,
                                                                                                 df_fitting_cycle_45,
                                                                                                 55,
                                                                                                 45,
                                                                                                 df_calculated_area_55C, 
                                                                                                 df_calculated_area_45C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        # Base 55C
        df_backcalculate_target_25_base_55 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_25,
                                                                                                 df_fitting_cycle_55,
                                                                                                 25,
                                                                                                 55,
                                                                                                 df_calculated_area_25C, 
                                                                                                 df_calculated_area_55C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        df_backcalculate_target_35_base_55 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_35,
                                                                                                 df_fitting_cycle_55,
                                                                                                 35,
                                                                                                 55,
                                                                                                 df_calculated_area_35C, 
                                                                                                 df_calculated_area_55C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        df_backcalculate_target_45_base_55 = calculate_equivalent_time_using_actual_data_notNorm_v2(df_fitting_cycle_45,
                                                                                                 df_fitting_cycle_55,
                                                                                                 45,
                                                                                                 55,
                                                                                                 df_calculated_area_45C, 
                                                                                                 df_calculated_area_55C,
                                                                                                 batch_name,
                                                                                                 desired_foldername)
        
        
        # Create dictionary for all the results
        df_backcalculation = {
            'batch_name':batch_name,
            'backcalculate_target_35_base_25': df_backcalculate_target_35_base_25,
            'backcalculate_target_45_base_25': df_backcalculate_target_45_base_25,
            'backcalculate_target_55_base_25': df_backcalculate_target_55_base_25,
            ######################################################################
            'backcalculate_target_25_base_35': df_backcalculate_target_25_base_35,
            'backcalculate_target_45_base_35': df_backcalculate_target_45_base_35,
            'backcalculate_target_55_base_35': df_backcalculate_target_55_base_35,
            ######################################################################
            'backcalculate_target_25_base_45': df_backcalculate_target_25_base_45,
            'backcalculate_target_35_base_45': df_backcalculate_target_35_base_45,
            'backcalculate_target_55_base_45': df_backcalculate_target_55_base_45,
            ######################################################################
            'backcalculate_target_25_base_55': df_backcalculate_target_25_base_55,
            'backcalculate_target_35_base_55': df_backcalculate_target_35_base_55,
            'backcalculate_target_45_base_55': df_backcalculate_target_45_base_55,
            }
        
        # Append to list
        list_all_backcalculations.append(df_backcalculation)
    
    return(list_all_backcalculations)


def run_calculate_equivalent_time_all_batches_cumSum_actualValue(batch_fitting_notNorm, desired_foldername): 
    
    list_all_backcalculations = []
    
    # Create folder if not exists yet
    if not os.path.exists('figures/'+folder_run_name+desired_foldername):
        os.makedirs('figures/'+folder_run_name+desired_foldername)
    
    # Go through all the lists
    for i in range(len(batchname_list)):
        
        batch_name = batch_fitting_notNorm[i][0]
        
        df_median_35C = batch_fitting_notNorm[i][4]
        df_median_45C = batch_fitting_notNorm[i][5]
        df_median_55C = batch_fitting_notNorm[i][6]
        
        # This calculates the 
        cumsum_median_35C = calculate_cumsum_median(df_median_35C)
        cumsum_median_45C = calculate_cumsum_median(df_median_45C)
        cumsum_median_55C = calculate_cumsum_median(df_median_55C)
        cumsum_median_55C_max = cumsum_median_max(cumsum_median_55C)
        cumsum_median_45C_max = cumsum_median_max(cumsum_median_45C)
        
        # Baseline: 35C, target: 55C
        df_analytical_55C_wrt_35C = find_cumsum_median_in_45C(cumsum_median_35C, cumsum_median_55C_max)
        
        # Baseline: 35C, target: 45C
        df_analytical_45C_wrt_35C = find_cumsum_median_in_45C(cumsum_median_35C, cumsum_median_45C_max)
        
        # Baseline: 45C, target: 55C
        df_analytical_55C_wrt_45C = find_cumsum_median_in_45C(cumsum_median_45C, cumsum_median_55C_max)
        
        # Put into the list
        list_all_backcalculations.append([batch_name,
                                          df_analytical_55C_wrt_35C,
                                          df_analytical_45C_wrt_35C,
                                          df_analytical_55C_wrt_45C])
        
        # Save as csv
        df_analytical_55C_wrt_35C.to_csv('figures/'+folder_run_name+desired_foldername+
                                         'analytical_55C_wrt_35C_actualValue_'+batch_name+'.csv',
                                         index=False)
        df_analytical_45C_wrt_35C.to_csv('figures/'+folder_run_name+desired_foldername+
                                         'analytical_45C_wrt_35C_actualValue_'+batch_name+'.csv',
                                         index=False)
        df_analytical_55C_wrt_45C.to_csv('figures/'+folder_run_name+desired_foldername+
                                         'analytical_55C_wrt_45C_actualValue_'+batch_name+'.csv',
                                         index=False)
        
        # Now plotting
        plot_equivalent_cumsum(df_analytical_55C_wrt_35C, 35, 55, batch_name, desired_foldername)
        plot_equivalent_cumsum(df_analytical_45C_wrt_35C, 35, 45, batch_name, desired_foldername)
        plot_equivalent_cumsum(df_analytical_55C_wrt_45C, 45, 55, batch_name, desired_foldername)
        
        
    return list_all_backcalculations
        

#%% Running code: loading all parameters

# SAM-based: Ulas, NiOx-based: Zahra-Batch2
# For the 25C
cell_params_25C = {0: [0, 'Zahra-Batch1'],
                   1: [1, 'Zahra-Batch2'], # NiOx-based
                   2: [1, 'Zahra-Batch2'], # NiOx-based
                   3: [0, 'Zahra-Batch1'],
                   4: [2, 'Ulas'], # SAM-based
                   5: [2, 'Ulas'], # SAM-based
                   6: [2, 'Ulas'], # SAM-based 
                   7: [2, 'Ulas']} # SAM-based

# For the 35C
cell_params_35C = {4: [0, 'Zahra-Batch1'],
                   5: [1, 'Zahra-Batch2'], # NiOx-based
                   6: [1, 'Zahra-Batch2'], # NiOx-based
                   7: [0, 'Zahra-Batch1'], 
                   0: [2, 'Ulas'], # SAM-based
                   1: [2, 'Ulas'], # SAM-based
                   2: [2, 'Ulas'], # SAM-based 
                   3: [2, 'Ulas']} # SAM-based

# For the 45C
cell_params_45C = {0 : [0,'Zahra-Batch1'],
                   1 : [1,'Zahra-Batch2'], # NiOx-based
                   2 : [1,'Zahra-Batch2'], # NiOx-based
                   3 : [0,'Zahra-Batch1'],
                   4 : [2,'Ulas'], # SAM-based
                   5 : [2,'Ulas'], # SAM-based
                   6 : [2,'Ulas'], # SAM-based
                   7 : [2,'Ulas']} # SAM-based

# For the 55C
cell_params_55C = {0 : [0,'Zahra-Batch1'],
                   1 : [1,'Zahra-Batch2'], # NiOx-based
                   2 : [1,'Zahra-Batch2'], # NiOx-based
                   3 : [0,'Zahra-Batch1'],
                   4 : [2,'Ulas'], # SAM-based
                   5 : [2,'Ulas'], # SAM-based
                   6 : [2,'Ulas'], # SAM-based
                   7 : [2,'Ulas']} # SAM-based


# path_25C = "/acceleratedcycle_script/dataset/20230203_130521_Acc_cyc_25C_12-12/extracted/MPPT_filtered/"
# path_35C = "/acceleratedcycle_script/dataset/20230602_152145_Acc_cyc_35C_6-6/extracted/MPPT_filtered/"
# path_45C = "/acceleratedcycle_script/dataset/20230320_123735_Acc_cyc_45C_3-3/extracted/MPPT_filtered/"
# path_55C = "/acceleratedcycle_script/dataset/20230320_123401_Acc_cyc_55C_1c5-1c5/extracted/MPPT_filtered/"

path_25C = "/acceleratedcycle_script/dataset/MPPT_filtered_25C/"
path_35C = "/acceleratedcycle_script/dataset/MPPT_filtered_35C/"
path_45C = "/acceleratedcycle_script/dataset/MPPT_filtered_45C/"
path_55C = "/acceleratedcycle_script/dataset/MPPT_filtered_55C/"

# Other parameters
limit_h_25C = 1200
limit_h_35C = 400
limit_h_45C = 1200
limit_h_55C = 1200

# Folder run name
folder_run_name = '/20240401_run/'

# Look at the current working directory
print('Current working directory: ',os.getcwd())

# Create folder if not exists yet
if not os.path.exists('figures/'+folder_run_name):
    os.makedirs('figures/'+folder_run_name)
    
sns.set_style(style=None)
    

#%% Running code: load data, calculate area per cycle & statistics

# Load data from the folders into df
df_all_25C, df_all_35C, df_all_45C, df_all_55C = run_load_data_python_allTs(path_25C, path_35C, path_45C, path_55C,
                                                                            cell_params_25C, cell_params_35C, cell_params_45C, cell_params_55C, 
                                                                            limit_h_25C, limit_h_35C, limit_h_45C, limit_h_55C)

# Calculate area per cycle
df_all_25C_area, df_all_35C_area, df_all_45C_area, df_all_55C_area, df_all_25C_35C_45C_55C_area =  run_calculate_area_allTs(df_all_25C, 
                                                                                                                           df_all_35C,
                                                                                                                           df_all_45C,
                                                                                                                           df_all_55C)
# Calculate area statistics
df_stats_25C, df_stats_35C, df_stats_45C, df_stats_55C, df_all_stats = run_calculate_area_stats_allTs(df_all_25C_area, df_all_35C_area, 
                                                                                                      df_all_45C_area, df_all_55C_area)

#%% Running code: look at the difference between 0 vs 1000 hours & plot comparison

df_0_1000_25C, df_0_1000_35C, df_0_1000_45C, df_0_1000_55C = run_calculate_0vs1000h_allTs(df_all_25C, df_all_35C, df_all_45C, df_all_55C)

plot_ratio_Vmpp_PCEmpp_0_1000(df_0_1000_25C, df_0_1000_35C, df_0_1000_45C,
                              df_0_1000_55C)


#%% Running code: plotting part 1 (area per cycle for various parameters and batches)

# Go through 2 specific types of params: 'normal' and 'loss'
params_type = ['normal', 'loss']
sns.set_style(style=None)

for specific_params_type in params_type:
    # Plot for a specific batch
    plot_area_parameters_specific_batch(df_all_25C_area, df_all_35C_area, df_all_45C_area,
                                    df_all_55C_area, folder_run_name, specific_params_type)
    
    # Plot for a specific batch, no 35C
    plot_area_parameters_specific_batch_no35C(df_all_25C_area, df_all_45C_area,
                                              df_all_55C_area, folder_run_name, specific_params_type)

    # Plot for all batches
    plot_area_parameters(df_all_25C_area, df_all_35C_area, df_all_45C_area,
                         df_all_55C_area, folder_run_name, specific_params_type)
    

# Plot stats (mean and std) for specific batch area
plot_area_parameters_stats_specific_batch(df_all_25C_area, df_all_35C_area, df_all_45C_area,
                                          df_all_55C_area, folder_run_name)


#%% Running code: regression of main parameters + arrhenius fitting

# Clear figures
plt.close()
sns.set_style(style=None)

# Go through for a specific T and specific batch
calculate_regression_specificT_specificBatch(df_all_25C_area, df_all_35C_area,
                                             df_all_45C_area, df_all_55C_area,
                                             'specific_batch')

calculate_regression_specificT_specificBatch(df_all_25C_area, df_all_35C_area,
                                             df_all_45C_area, df_all_55C_area,
                                             'all_batches')

# Regression for fitting each pixel
df_fitting = calculate_regression_eachpixel(df_all_25C_35C_45C_55C_area)

# Calculate median, fit arrhenius
calculate_median_slope_arrhenius(df_fitting)

# Calculate median, fit arrhenius with unifor ylim
calculate_median_slope_arrhenius_ylim(df_fitting)

# Doing regression on arrhenius plot
df_regression_arrhenius = regression_arrhenius(df_fitting)


#%% Running code: find comparable area across different T for specific batch

sns.set_style(style=None)

# For normalized
# batch_fitting = fit_aging_cycle_function(df_all_35C, df_all_45C, df_all_55C)

# For not normalized
batch_fitting_notNorm = fit_aging_cycle_function_notNorm(df_all_25C, df_all_35C, df_all_45C, df_all_55C)


#%% Running code: read all the dfs, decide which one is the best

folder_path = 'figures/'+folder_run_name+'cycle_fitting_notNorm/'

# Create an empty dictionary to store DataFrames
dfs = {}

# Iterate through all files in the folder
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        # Read the CSV file and store its contents as a pandas DataFrame
        df_name = os.path.splitext(file)[0]  # Extract filename without extension
        dfs[df_name] = pd.read_csv(os.path.join(folder_path, file))

# Assuming you already have the dictionary dfs
for df_name, df in dfs.items():
    r_squared_columns = [col for col in df.columns if col.startswith('r_squared')]
    if r_squared_columns:
        r_squared_subset = df[r_squared_columns].replace([-np.inf, np.nan], np.nan)
        filtered_subset = r_squared_subset.dropna()
        means = filtered_subset.mean()
        print(f"DataFrame: {df_name}")
        print("Means of columns starting with 'r_squared' (excluding -inf and NaN):")
        print(means)
        print('\n')
    else:
        print(f"No columns starting with 'r_squared' in DataFrame {df_name}")

# Find the mean
result_dict = {}

for df_name, df in dfs.items():
    r_squared_columns = [col for col in df.columns if col.startswith('r_squared')]
    if r_squared_columns:
        r_squared_subset = df[r_squared_columns].replace([-np.inf, np.nan], np.nan)
        filtered_subset = r_squared_subset.dropna()
        means = filtered_subset.mean()
        result_dict[df_name] = means

result_df = pd.DataFrame(result_dict)

# Find the index with the highest value for each column
highest_indices = result_df.idxmax()

# Create a new row with the highest indices for each column
result_df.loc["highest_index"] = highest_indices
        
#%% Running code: all backcalculations (using fitted equation)

sns.set_style(style=None)
# list_all_backcalculations = run_calculate_equivalent_time_all_batches(batch_fitting, 'backcalculate/')
# list_all_backcalculations_notNorm = run_calculate_equivalent_time_all_batches_notNorm(batch_fitting_notNorm, 'backcalculate_notNorm/')

list_all_backcalculations_notNorm = run_calculate_equivalent_time_all_batches_notNorm_v2(batch_fitting_notNorm, 'backcalculate_notNorm/')

        
#%% Running code: plot the boxplot of the backcalculation not norm  FOR 25, 45, 55C + FITTING THE MEDIAN

# Define the exponential function
def exp_func_T_eqTime(x, a, b):
    return a * np.exp(b*x)

# Define the exp function fit
def exp_fit_T_eqTime(x, y):
    
#     # Remove non-positive values from the data
#     mask = y > 0
#     x_fit = x[mask]
#     y_fit = y[mask]
    
    # Give initial guess of the parameters
    initial_guess = [30,-0.0598]
    
    # Perform the curve fitting
    popt, _ = curve_fit(exp_func_T_eqTime, x, y, p0=initial_guess,maxfev=10000)

    # Extract the fitted parameters
    a_fit, b_fit = popt

    return a_fit, b_fit

# Choose a palette color
# sns.set(style="whitegrid")
sns.set_theme() 
palette = sns.color_palette("deep", 3)
palette_dict = {'Ulas':palette[0],
                'Zahra-Batch1':palette[1],
                'Zahra-Batch2':palette[2]}

length_base = {25:12,45:3,55:1.5}#35:6,

desired_foldername='backcalculate_notNorm_exc35C/'
os.makedirs("figures/"+folder_run_name+desired_foldername, exist_ok=True)

# List to add
batch_name_list = []
base_T_list = []
a_fit_list = []
b_fit_list = []
r_squared_list = []

for i in range(len(list_all_backcalculations_notNorm)):
    
    specific_batch_df = list_all_backcalculations_notNorm[i]
    
    # Filter out anything containing 35
    specific_batch_df = {key: value for key, value in specific_batch_df.items() if 'target_35' not in key and 'base_35' not in key}
    
    batch_name = specific_batch_df['batch_name']
    print(batch_name)
    
    # List all the temperatures
    base_T_C = [25,45,55] #35
    base_T_K = [x + 273 for x in base_T_C]
    
    
    for base_T in base_T_C:
        
        try:
        
            # List to store keys ending with "_base_<T>"
            keys_ending_with_base_T = [key for key in specific_batch_df.keys() if key.endswith('_base_'+str(base_T))]
            print(keys_ending_with_base_T)
            
            # Find temperatures not in base_T     
            # Create a new list containing elements after the selected element
            non_base_T_C = [x for x in base_T_C if x != base_T]
            print(non_base_T_C)
            
            # Create a dataframe
            df_interest_plot = pd.DataFrame(columns=['eq_time', 'T_C'])
            df_interest_plot.loc[0] = [length_base[base_T],base_T] # Add the base temperature
            
            # Going through each T and calculate the equivalent time
            for key_ending_base_T,non_base_T_C_specific in zip(keys_ending_with_base_T,non_base_T_C):
                specific_base_T_df = specific_batch_df[key_ending_base_T]#[0]
                
                print(key_ending_base_T,non_base_T_C_specific,base_T)
                column_interest = 'eq_'+str(non_base_T_C_specific)+'C_wrt_'+str(base_T)+'C_time_linear_log'
                
                df_interest_plot_specific=(specific_base_T_df[0].to_frame()).dropna()
                df_interest_plot_specific = df_interest_plot_specific.rename(columns={column_interest:'eq_time'})
                df_interest_plot_specific['T_C']=non_base_T_C_specific
                
                df_interest_plot = pd.concat([df_interest_plot, df_interest_plot_specific], ignore_index=True)
                
            # Now trying to fit
            
            # Finding the median of 'eq_time' for each unique value of 'T_C'
            median_data = df_interest_plot.groupby('T_C')['eq_time'].median().reset_index()
            percentile_25th = df_interest_plot.groupby('T_C')['eq_time'].quantile(0.25).reset_index()
            percentile_75th = df_interest_plot.groupby('T_C')['eq_time'].quantile(0.75).reset_index()
            
            # Creating a new DataFrame with the median values
            median_data_df = pd.DataFrame(median_data, columns=['T_C', 'eq_time'])
            
            # Fitting the exponential decay function to the data
            a_fit, b_fit = exp_fit_T_eqTime(median_data_df['T_C'], median_data_df['eq_time'])
            
            # Generating the fitted curve
            x_fit = np.linspace(20,60,100)
            y_fit = exp_func_T_eqTime(x_fit, a_fit, b_fit)
            
            x_fit_short = np.array((25,45,55)) #35
            y_fit_short = exp_func_T_eqTime(x_fit_short, a_fit, b_fit)
            
            
            # Calculating the total sum y_fit squares and residual sum of squares
            ss_res = np.sum(( median_data_df['eq_time'] - y_fit_short) ** 2)
            ss_tot = np.sum(( median_data_df['eq_time'] - np.mean(y_fit_short)) ** 2)
            
            # # Calculating R-squared
            r_squared_calc = 1 - (ss_res / ss_tot)
            
            # Start plotting figure
            # plt.close()
            plt.figure(figsize=(2.5,3),dpi=300)
            # fig, ax = plt.subplots(figsize=(4,3),dpi=300)
            
            sns.scatterplot(data=median_data_df,x='T_C',y='eq_time')
            sns.lineplot(x=x_fit,y=y_fit, color='red',label='Exponential fit')
            
            # plt.title(f'{a_fit}*exp(-{b_fit}x)')#', r_squared: {r_squared}')
            
            # Extract values from percentile DataFrames
            lower_error = median_data_df['eq_time'] - percentile_25th['eq_time']
            upper_error = percentile_75th['eq_time'] - median_data_df['eq_time']
            
            # Adding error bars
            plt.errorbar(median_data_df['T_C'], median_data_df['eq_time'], yerr=[lower_error, upper_error], fmt='o', capsize=3, capthick=1, elinewidth=1)

            plt.legend().remove()            
            plt.tight_layout()
            
            plt.savefig("figures/"+folder_run_name+desired_foldername+"scatterplot_temperature_equivalent_notNorm_"+
                        batch_name+"_base_"+str(base_T)+".png",
                        dpi=300)
            
            # Add list
            batch_name_list.append(batch_name)
            base_T_list.append(base_T)
            a_fit_list.append(a_fit)
            b_fit_list.append(b_fit)
            r_squared_list.append(r_squared_calc)
            
            
            
        except:
            print(f'{column_interest} error!')
            break

# Create dataframe from all the data
dict_df = {
    'batch_name': batch_name_list,
    'base_T_C': base_T_list,
    'a_fit': a_fit_list,
    'b_fit': b_fit_list,
    'r_squared': r_squared_list,
    }

result_fitting_eq_time_df = pd.DataFrame(dict_df)

# Now calculate for 25C vs 55C, and acceleration factors?
result_fitting_eq_time_df['25C_eq_time']=result_fitting_eq_time_df['a_fit']*np.exp(25*result_fitting_eq_time_df['b_fit'])
result_fitting_eq_time_df['55C_eq_time']=result_fitting_eq_time_df['a_fit']*np.exp(55*result_fitting_eq_time_df['b_fit'])
result_fitting_eq_time_df['acceleration_factor_25C_55C']=result_fitting_eq_time_df['25C_eq_time']/result_fitting_eq_time_df['55C_eq_time']

# Save to csv
result_fitting_eq_time_df.to_csv("figures/"+folder_run_name+desired_foldername+
                                 "scatterplot_temperature_equivalent_notNorm_fitting_exponential.csv",
                          index=False)

        