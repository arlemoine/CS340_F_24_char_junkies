'''
#%% MODULE BEGINS
module_name = 'config'

Version: 1.0.0

Description:
    This module holds configuration constants for HRV calculations, step score calculations, max department size, file names and file paths.

Authors:
    Chris Smith
    Adriean Lemoine

Date Created     :  11/12/24
Date Last Updated:  11/14/24
'''

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Data access paths
DIR_INPUT = 'Input/'  # This is the path where we'll access our input data files
DIR_OUTPUT = 'Output/'  # This is the path where we'll save our output files

# Filenames 
FILENAME_STEPS = 'steps.csv' # Filename for steps data 
FILENAME_AGE = 'age.csv' # Filename for age data 

# Data Cleaning Constants   
MAX_HEART_RATE = 120  # We will use this threshold to identify high-stress periods in HRV data based on heart rate

# Define max department size for police officers
MAX_DEPARTMENT_SIZE = 100  # Example maximum size for the police department

# Fitness Score Constants
STEP_WEIGHT = 0.3  # This weight is applied to the step score
HRV_WEIGHT = 0.7  # This weight is applied to the HRV score
MIN_FITNESS_SCORE = 1600 # Minimum fitness score of an individual to avoid probation