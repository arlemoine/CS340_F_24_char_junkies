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
import csv
import seaborn as sns

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import Config

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Class definitions Start Here

class FitnessData:
    """
    Parent class for managing fitness data of an individual.
    """
    
    # Initialize the FitnessData object with personal and fitness data
    def __init__(self, name):
        self.name = name  # Store person's name

        self.config = {
            "DIR_INPUT": Config.DIR_INPUT,
            "FILENAME_STEPS": Config.FILENAME_STEPS,
            "FILENAME_AGE": Config.FILENAME_AGE,
            "MAX_HEART_RATE": Config.MAX_HEART_RATE,
            "HRV_WEIGHT": Config.HRV_WEIGHT,
            "STEP_WEIGHT": Config.STEP_WEIGHT
        }

        self.logger = logging.getLogger(__name__)
        self.configureLogger()
        self.logger.info('Person created.')

        self.age = None
        self.df_steps = None
        self.df_hrv = None 
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

    def view_table(self, dataframe):
        if dataframe is not None:
            print(dataframe.head()) 
        else:
            print("Dataframe is empty.")
    #

    # # Display the dataframe as a table
    def view_table(self, dataframe):
        if dataframe is not None:
            print(dataframe.head())  # Display the first few rows of the dataframe #
        else:
            print("Dataframe is empty.")  # Print a message if the dataframe is empty #
    

    # # Plot a line graph for a specific column in the dataframe
    # def view_line_graph(self, dataframe, column, title):
    #     if dataframe is not None and column in dataframe.columns:
    #         dataframe.plot(y=column, kind='line', title=title)  # Plot the line graph #
    #     else:
    #         print("Column not found in the dataframe or dataframe is empty.")  # Print a message if the column is not found #
    # 

    def view_line_graph(self, dataframe, column, title):
        if dataframe is not None and column in dataframe.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(dataframe['dayOfMonth'], dataframe[column], marker='o', linestyle='-')
            personal_title = f"{title} - {self.name.capitalize()}"
            plt.title(personal_title)
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.grid(True)
            plt.show()
            self.logger.info(f"Displayed line graph for {column} in {title}.")
        else:
            self.logger.warning(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
            print(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
    #

    # # Display the steps dataframe as a table
    def view_steps_table(self):
        self.view_table(self.df_steps) # Use the view_table method to display steps dataframe #

    # # Display the HRV dataframe as a table
    def view_hrv_table(self):
        self.view_table(self.df_hrv)  # Use the view_table method to display HRV dataframe #
    # 

    # # Plot a line graph of steps data
    def view_steps_line_graph(self):
        self.view_line_graph(self.df_steps, 'steps', 'Steps Over Time')  # Plot line graph for steps #
    # #

        # Plot a line graph of HRV data
    def view_hrv_line_graph(self):
        self.view_line_graph(self.df_hrv, 'hrv', 'HRV Over Time')  # Plot line graph for HRV #
    # 
#
    def data_search(self, dataframe, column, search_value):
        ##(dataframe) used to declare which dataframe to search in
        ##(column) used to declare which column to search in
        ##(search_value) used to declare the value we are searching for. Takes in str,int,or float
        
        if dataframe is not None and column in dataframe.columns:
            # Search for the value in the given column and return the matching rows
            result = dataframe[dataframe[column] == search_value]
            if not result.empty:
                print(f"Here are the results in column '{column}' for value '{search_value}':")
                print(result)
                self.logger.info(f"Search results for '{search_value}' in column '{column}':")
                self.logger.info(result)
                return result#returns the row from the dataframe that match the search value
            else:
                print(f"No matching results found for '{search_value}' in column '{column}'.")
                self.logger.warning(f"No matching results found for '{search_value}' in column '{column}'.")
                return None
        else:
            print(f"The dataframe is empty or column '{column}' was not found.")
            self.logger.warning(f"The dataframe is empty or column '{column}' was not found.")
            return None

class FitnessDataProcessing(FitnessData):
    '''
    Child class for processing and analyzing fitness data.
    '''
    
    # Initialize the FitnessDataProcessing object with additional attributes
    def __init__(self, name):
        super().__init__(name)
        
    #

    # Import age from age.csv
    def importAge(self):
        filepath = self.config["DIR_INPUT"] + self.name + '/age.csv'

        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader) # To skip the header row

                for row in reader:
                    self.age = int(row[1])
                
                self.logger.info(f'Age ({self.age}) extracted from {filepath}')

            # temp = pd.read_csv(filepath)  # Read the CSV file into a temporary dataframe
            # self.logger.info(f"Successfully imported {filepath}")  # Print success message #
            # return df  # Return the dataframe #
        except FileNotFoundError:
            print(f"File {filepath} not found.")  # Print error message if file is not found #
            return None  # Return None if file is not found #
    #

    # Import steps.csv into dataframe
    def importSteps(self):
        filepath = self.config["DIR_INPUT"] + self.name + '/steps.csv'

        try:
            with open(filepath, 'r') as csvfile:
                self.df_steps = pd.read_csv(filepath)  # Read the CSV file into dataframe ##changed pd to df
                self.logger.info(f'Step data is loaded.')

        except FileNotFoundError:
            print(f"{filepath} not found.")  # Print error message if file is not found
    #

    def importHrv(self): ## Attempt at hrv import
        filePath = self.config["DIR_INPUT"] + self.name + '/hrv/'
        try: 
            hrv_files = [f'{filePath}{file}' for file in os.listdir(filePath) if file.endswith('.csv')]
            self.df_hrv = pd.DataFrame(columns=['name','dayOfMonth','hrv']) # Create empty df for avg hrv per day

            for file in hrv_files:
                with open(file, 'r') as csvfile:
                    rawData = pd.read_csv(csvfile) # Pull hrvDD.csv into df
                    cleanData = rawData[rawData['heartRate'] < self.config['MAX_HEART_RATE']] # Remove high-stress records
                    avg_hrv = round(cleanData['hrv'].mean()) # Calculate avg hrv for current day
                    dom = int(file[-6:-4]) # Grab day of month from file name
                    new_row = pd.DataFrame({'name':[self.name], 'dayOfMonth':[dom], 'hrv':[avg_hrv]}) # Create new row
                    self.df_hrv = pd.concat([self.df_hrv,new_row], ignore_index=True)

                    self.logger.info(f"HRV data loaded from {file}")                
        except FileNotFoundError:
            print(f"{filePath} not found.")
    
    
    def visualize_violin_plot(self, dataframe, column, title="Violin Plot"):
        ## Visualizes a violin plot for a specified column in the specified dataFrame
        
        ## (dataframe) for the dataframe we wish to plot
        ## (column) takes in a str for the column for which we wish to plot
        ## (title) the title of the violin plot

        ## Check if dataframe has data and the specified column exists
        if dataframe is not None and column in dataframe.columns:
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=dataframe, x=column)
            plt.title(f"{title} - {self.name.capitalize()}")
            plt.xlabel(f"{column} per month")
            plt.ylabel('Density')
            plt.grid(True)
            plt.show()
            self.logger.info(f"Displayed violin plot for {column} in {title}.")
        else:
            ## If dateframe is empty or the column is not found it is logged and an error message is printed
            self.logger.warning(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
            print(f"Column '{column}' not found in DataFrame or DataFrame is empty.")


    def query_data(self, dataframe, query_column, condition, value):
        ## Queries the specified dataframe based on the given condition and value for the specified column

        ## (dataframe) for the dataframe we are searching in
        ## (query_column) takes in a str for the column to apply the condition to
        ## (condition) takes in a str for the condition to apply. Str can be one of ['==', '>', '<', '>=', '<=', '!=']
        ## (value) takes in an int, float, or str for the value to compare against
        
        ## Check if dataframe has data and the specified column exists
        if dataframe is not None and query_column in dataframe.columns:
            if condition == '==':
                result = dataframe[dataframe[query_column] == value]
            elif condition == '>':
                result = dataframe[dataframe[query_column] > value]
            elif condition == '<':
                result = dataframe[dataframe[query_column] < value]
            elif condition == '>=':
                result = dataframe[dataframe[query_column] >= value]
            elif condition == '<=':
                result = dataframe[dataframe[query_column] <= value]
            elif condition == '!=':
                result = dataframe[dataframe[query_column] != value]
            else:
                ## Check for invalid condition
                self.logger.error(f"Invalid condition: {condition}")
                return None
        
            ## Check if any rows match the query condition
            if not result.empty:
                print(f"Query results for {query_column} {condition} {value}:")
                print(result)
                self.logger.info(f"Query results for {query_column} {condition} {value}:")
                self.logger.info(result)  # Log the query result
                return result  # Return the matching rows
            else:
                ## Notify user if no rows match the condition
                print(f"No matching results found for {query_column} {condition} {value}.")
                self.logger.warning(f"No matching results found for {query_column} {condition} {value}.")
                return None
        else:
            ## Notify user if the dataframe is empty
            print(f"The dataframe is empty or column '{query_column}' was not found.")
            self.logger.warning(f"The dataframe is empty or column '{query_column}' was not found.")
            return None

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
    
    pers1 = FitnessDataProcessing('adam')
    pers1.importAge()
    pers1.importSteps()
    pers1.importHrv()

    pers1.view_steps_table()
    pers1.view_hrv_table()

    pers1.view_steps_line_graph()
    pers1.view_hrv_line_graph()

    pers1.data_search(pers1.df_steps, 'dayOfMonth', 3)
    pers1.data_search(pers1.df_hrv, 'hrv', 16)  

    pers1.visualize_violin_plot(pers1.df_steps, 'steps', 'Violin Plot of Steps')
    pers1.visualize_violin_plot(pers1.df_hrv, 'hrv', 'Violin Plot of HRV')

    
    result = pers1.query_data(pers1.df_steps, 'steps', '>', 5000)
    result = pers1.query_data(pers1.df_steps, 'dayOfMonth', '==', 31)


    '''
    pers2 = FitnessDataProcessing('brian')
    pers2.importAge()
    pers2.importSteps()
    pers2.importHrv()

    pers2.view_steps_table()
    pers2.view_hrv_table()

    pers2.view_steps_line_graph()
    pers2.view_hrv_line_graph()

    pers2.data_search(pers1.df_steps, 'dayOfMonth', 3) 
    pers2.data_search(pers1.df_hrv, 'hrv', '18') 
    '''
    
    #TEST Code
    # main()

