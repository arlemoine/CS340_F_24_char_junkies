module_name = 'Individuals'

'''
Version: <V1>

Description:
    <This module is to gather data for each individual officer, process the data, and create data frames using the collected data>

Authors:
    Chris Smith
    Adriean Lemoine

Date Created:       11/12/24
Date Last Updated:  11/14/24
'''
#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

from matplotlib import pyplot as plt
import numpy  as np 
import pandas as pd
import logging

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import Config

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Class definitions Start Here

class FitnessData:
    """
    Parent class for managing fitness data of an individual.
    """

    # The following will be filled when pulling csv into dataframe
    df_steps = None
    df_hrv = None 
    age = None
    
    # Initialize the FitnessData object with personal and fitness data
    def __init__(self, name):
        self.name = name  # Store person's name

        self.logger = logging.getLogger(__name__)
        self.configureLogger()
        self.logger.info('Person created.')
    #

    # Configure logging functionality of object
    def configureLogger(self):
        # Ensure directory exists
        log_dir = 'Log'
        os.makedirs(log_dir, exist_ok=True)

        # Set logger configurations
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{self.name}.log"))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)  

        # Attach handler to logger
        self.logger.addHandler(file_handler)
    #

    def print(self):
        print(f'Name:\t\t{self.name}')
        print(f'Age:\t\t{self.age}')
        print(f'Steps Data:\t{self.df_steps}')
        print(f'HRV Data:\t{self.df_hrv}')
    #
    
    # # Display the dataframe as a table
    # def view_table(self, dataframe):
    #     if dataframe is not None:
    #         print(dataframe.head())  # Display the first few rows of the dataframe #
    #     else:
    #         print("Dataframe is empty.")  # Print a message if the dataframe is empty #
    # #

    # # Plot a line graph for a specific column in the dataframe
    # def view_line_graph(self, dataframe, column, title):
    #     if dataframe is not None and column in dataframe.columns:
    #         dataframe.plot(y=column, kind='line', title=title)  # Plot the line graph #
    #     else:
    #         print("Column not found in the dataframe or dataframe is empty.")  # Print a message if the column is not found #
    # #

    # # Display the steps dataframe as a table
    # def view_steps_table(self):
    #     self.view_table(self.df_steps)  # Use the view_table method to display steps dataframe #
    # #

    # # Display the HRV dataframe as a table
    # def view_hrv_table(self):
    #     self.view_table(self.df_hrv)  # Use the view_table method to display HRV dataframe #
    # #

    # # Plot a line graph of steps data
    # def view_steps_line_graph(self):
    #     self.view_line_graph(self.df_steps, 'Steps', 'Steps Over Time')  # Plot line graph for steps #
    # #

    # # Plot a line graph of HRV data
    # def view_hrv_line_graph(self):
    #     self.view_line_graph(self.df_hrv, 'HRV', 'HRV Over Time')  # Plot line graph for HRV #
    # #
#

class FitnessDataProcessing(FitnessData):
    '''
    Child class for processing and analyzing fitness data.
    '''
    
    # Initialize the FitnessDataProcessing object with additional attributes
    def __init__(self, name):
        super().__init__(name)
        
        
    #

    # Import CSV files into dataframes
    def import_csv(self, filename):
        try:
            df = pd.read_csv(filename)  # Read the CSV file into a dataframe #
            print(f"Successfully imported {filename}")  # Print success message #
            return df  # Return the dataframe #
        except FileNotFoundError:
            print(f"File {filename} not found.")  # Print error message if file is not found #
            return None  # Return None if file is not found #
    #

    # # Remove bad HRV data when heart rate > max
    # def clean_data(self):
    #     self.df_hrv = self.df_hrv[self.df_hrv['HeartRate'] <= self.CONSTANTS['MAX_HEART_RATE']]  # Filter out rows where heart rate exceeds the max #
    # #

    # # Create dataframe for average HRV per day, age corrected
    # def calc_hrv(self):
    #     if not self.df_hrv.empty:
    #         self.df_hrv['CorrectedHRV'] = self.df_hrv['HRV'] * (10 / self.person_age)  # Apply age correction to HRV #
    #         self.df_average_hrv = self.df_hrv.groupby('Date')['CorrectedHRV'].mean().reset_index()  # Calculate daily average HRV #
    # #

    # # Calculate step score
    # def calc_step_score(self):
    #     if not self.df_steps.empty:
    #         self.avg_steps_per_day = self.df_steps['Steps'].mean()  # Calculate average steps per day #
    #         step_score = self.avg_steps_per_day * self.CONSTANTS['STEPS_WEIGHT']  # Calculate step score based on weight #
    #         return step_score  # Return the step score #
    # #

    # # Calculate HRV score
    # def calc_hrv_score(self):
    #     if not self.df_average_hrv.empty:
    #         self.avg_hrv_per_day = self.df_average_hrv['CorrectedHRV'].mean()  # Calculate average HRV per day #
    #         hrv_score = self.avg_hrv_per_day * self.CONSTANTS['HRV_WEIGHT']  # Calculate HRV score based on weight #
    #         return hrv_score  # Return the HRV score #
    # #

    # # Calculate fitness score
    # def calc_fitness_score(self):
    #     step_score = self.calc_step_score()  # Get step score #
    #     hrv_score = self.calc_hrv_score()  # Get HRV score #
    #     if step_score is not None and hrv_score is not None:
    #         self.fitness_score = step_score + hrv_score  # Combine step and HRV scores for overall fitness score #
    # #

    # # Graph HRV per day (whisker plot)
    # def graph_hrv_per_day(self):
    #     if not self.df_average_hrv.empty:
    #         self.df_average_hrv.boxplot(column='CorrectedHRV', by='Date')  # Create whisker plot for HRV per day #
    # #

    # # Graph steps per day (whisker plot)
    # def graph_steps_per_day(self):
    #     if not self.df_steps.empty:
    #         self.df_steps.boxplot(column='Steps', by='Date')  # Create whisker plot for steps per day #
    # #

    # # Show stats for the month
    # def show_stats_for_month(self):
    #     print(f"Average Steps per Day: {self.avg_steps_per_day}")  # Print average steps per day #
    #     print(f"Average HRV per Day: {self.avg_hrv_per_day}")  # Print average HRV per day #
    #     print(f"Fitness Score: {self.fitness_score}")  # Print fitness score #
    # #
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Main Self-run block
if __name__ == "__main__":
    
    pers1 = FitnessDataProcessing('bob')

    
    #TEST Code
    # main()

