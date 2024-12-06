module_name = 'Individuals'

'''
Version: 1.2

Description:
    <This module is to gather data for each individual officer, process the data, and create data frames using the collected data>

Authors:
    Chris Smith
    Adriean Lemoine
    Nicholas Burgo

Date Created:       11/12/2024
Date Last Updated:  12/05/2024
'''
#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    import os
    #os.chdir("./../..")
#

#custom imports
import Logging

#other imports
import contextlib as clib
import csv
from matplotlib import pyplot as plt
import numpy  as np 
import os
import pandas as pd
import seaborn as sns

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import Config

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Class definitions Start Here

class FitnessData:
    """
    Parent class for managing fitness data of an individual.
    """
    config = {
        "DIR_INPUT": Config.DIR_INPUT,
        "DIR_OUTPUT": Config.DIR_OUTPUT,
        "FILENAME_STEPS": Config.FILENAME_STEPS,
        "FILENAME_AGE": Config.FILENAME_AGE,
        "MAX_HEART_RATE": Config.MAX_HEART_RATE,
        "HRV_WEIGHT": Config.HRV_WEIGHT,
        "STEP_WEIGHT": Config.STEP_WEIGHT
    }

    age = None
    df_steps = None
    df_hrv = None
    departureAngle = None # Used in comparing individual's fitness data as a deviation from the department average fitness data

    # Initialize the FitnessData object with personal and fitness data
    def __init__(self, name):
        self.name = name  # Store person's name
        self.initLog()
    #

    # Generate a log file for the given individual
    def initLog(self):
        self.logger = Logging.configure_logger(self.name, f'Output/{self.name}/')
    #

    # Display the dataframe as a table
    def view_table(self, dataframe):
        pd.set_option('display.max_rows', None)
        if dataframe is not None:
            print(dataframe)  # Display the first few rows of the dataframe #
        else:
            print("Dataframe is empty.")  # Print a message if the dataframe is empty #
        #
    #

    # Display the steps dataframe as a table
    def view_steps_table(self):
        self.view_table(self.df_steps) # Use the view_table method to display steps dataframe #
    #

    # Display the HRV dataframe as a table
    def view_hrv_table(self):
        self.view_table(self.df_hrv)  # Use the view_table method to display HRV dataframe #
    # 

    # Plot a line graph for a specific column in the dataframe
    def get_line_graph(self, dataframe, column, title):
        if dataframe is not None and column in dataframe.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(dataframe['dayOfMonth'], dataframe[column], marker='o', linestyle='-')
            personal_title = f"{title} - {self.name.capitalize()}"
            plt.title(personal_title)
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.grid(True)
            plt.savefig(f"Output/{self.name}/line_{column}.png")
            plt.close()
            self.logger.info(f"Displayed line graph for {column} in {title}.")
        else:
            self.logger.warning(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
            print(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
        #
    #

    # Plot a line graph of steps data
    def get_steps_line_graph(self):
        self.get_line_graph(self.df_steps, 'steps', 'Steps Over Time')  # Plot line graph for steps #
    #

    # Plot a line graph of HRV data
    def get_hrv_line_graph(self):
        self.get_line_graph(self.df_hrv, 'hrv', 'HRV Over Time')  # Plot line graph for HRV #
    # 

    # Perform data search
    def data_search(self, dataframe, column, search_value):
        '''
        (dataframe) used to declare which dataframe to search in
        (column) used to declare which column to search in
        (search_value) used to declare the value we are searching for. Takes in str,int,or float
        '''

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
        #
    #

    # Print attributes of individual
    def print(self, *args):
        if not args:
            for attr in self.__dict__:
                print(f"{attr}: {getattr(self,attr)}")
        else:
            for arg in args:
                if hasattr(self, arg):
                    print(f"{arg}: {getattr(self, arg)}")
                else:
                    print(f"Attribute '{arg}' not found.")
                #
            #
        #   
    #
#

class FitnessDataProcessing(FitnessData):
    '''
    Child class for processing and analyzing fitness data.
    '''
    dirOut = "Output/" # Default output if no output is provided
    step_score = None
    hrv_score = None
    fitness_score = None

    # Initialize the FitnessDataProcessing object with additional attributes
    def __init__(self, name):
        super().__init__(name)
        self.dirOut = self.config["DIR_OUTPUT"] + self.name + '/' # Define output directory
        self.importAll()
        self.calc_all()
        self.writeOutputs()
        self.logger.info(f'Person \'{self.name}\' updated.')
    #

    # Used to write output file structure when it doesn't exist
    def writeOutputs(self):
        self.updateDirectory()
        self.writeAllToFile()
        self.genPlots()
        self.logger.info("Output files configured.")
    #

    # Create folder for person in Output folder
    def updateDirectory(self):
        if not os.path.exists(self.dirOut):
            os.makedirs(self.dirOut)
        #
    #

    # Import all data into dataframes from Input folder
    def importAll(self):
        self.importAge()
        self.importHrv()
        self.importSteps()
    #

    # Import age from age.csv
    def importAge(self):
        filepath = self.config["DIR_INPUT"] + self.name + '/' + self.config["FILENAME_AGE"]

        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader) # To skip the header row

                for row in reader:
                    self.age = int(row[1])
                #
                
                self.logger.info(f'Age ({self.age}) extracted from {filepath}')
            #
        except FileNotFoundError:
            print(f"File {filepath} not found.")  # Print error message if file is not found #
            return None  # Return None if file is not found
        #
    #

    # Import steps.csv into dataframe
    def importSteps(self):
        filepath = self.config["DIR_INPUT"] + self.name + '/' + self.config['FILENAME_STEPS']

        try:
            with open(filepath, 'r') as csvfile:
                self.df_steps = pd.read_csv(filepath)  # Read the CSV file into dataframe ##changed pd to df
                self.logger.info(f'Step data is loaded.')
        except FileNotFoundError:
            print(f"{filepath} not found.")  # Print error message if file is not found
        #
    #

    # Import hrv data into dataframe
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
        #
    #
    
    # Calculate all individual stats
    def calc_all(self):
        self.calc_step_score()
        self.calc_hrv_score()
        self.calc_fitness_score()
    #

    # Calculate step score
    def calc_step_score(self):
        if not self.df_steps.empty:
            meanSteps = "int(self.df_steps['steps'].mean())"
            self.avg_steps = eval(meanSteps)  # Calculate average steps per day
            self.step_score = self.avg_steps * self.config['STEP_WEIGHT'] / 100  # Calculate step score based on weight #
        #
    #

    # Calculate HRV score
    def calc_hrv_score(self):
        if not self.df_hrv.empty:
            self.avg_hrv = int(self.df_hrv['hrv'].mean())  # Calculate average HRV per day
            self.hrv_score = self.avg_hrv * self.config['HRV_WEIGHT']  # Calculate HRV score based on weight
        #
    #

    # Calculate fitness score
    def calc_fitness_score(self):
        if self.step_score is not None and self.hrv_score is not None:
            self.fitness_score = int(self.step_score + self.hrv_score)  # Combine step and HRV scores for overall fitness score
        #
    #

    # Write all data for individual to data.txt
    def writeAllToFile(self):
        dataFile = f'{self.dirOut}data.txt'

        with open(dataFile, "w") as f:
            with clib.redirect_stdout(f):
                print(f'Name:\t{self.name}')
                print(f'Age:\t{self.age}')
                print()
                self.show_stats_for_month()
                print('\nSTEPS PER DAY')
                self.view_steps_table()
                print('\nHRV PER DAY')
                self.view_hrv_table()
            #
        #
    #

    # Generate all plots for individual
    def genPlots(self):
        self.get_violin_plot(self.df_steps,'steps')
        self.get_violin_plot(self.df_hrv,'hrv')
        self.get_steps_line_graph()
        self.get_hrv_line_graph()
    #

    # Create violin plots
    def get_violin_plot(self, dataframe, column, title="Violin Plot"):
        '''
        Visualizes a violin plot for a specified column in the specified dataFrame
        
        (dataframe) for the dataframe we wish to plot
        (column) takes in a str for the column for which we wish to plot
        (title) the title of the violin plot
        '''

        # Check if dataframe has data and the specified column exists
        if dataframe is not None and column in dataframe.columns:
            # Configure plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=dataframe, x=column)
            plt.title(f"{title} - {self.name.capitalize()}")
            plt.xlabel(f"{column} per month")
            plt.ylabel('Density')
            plt.grid(True)  
            
            # Save violin plot to the output folder
            path = f"{self.config['DIR_OUTPUT']}{self.name}/"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(f'{path}violin_{column}.png')
            plt.close()
            self.logger.info(f'Violin plot for {column} saved to {path}')
        else:
            # If dateframe is empty or the column is not found it is logged and an error message is printed
            self.logger.warning(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
            print(f"Column '{column}' not found in DataFrame or DataFrame is empty.")
        #
    #

    def query_data(self, dataframe, query_column, condition, value):
        '''
        Queries the specified dataframe based on the given condition and value for the specified column

        (dataframe) for the dataframe we are searching in
        (query_column) takes in a str for the column to apply the condition to
        (condition) takes in a str for the condition to apply. Str can be one of ['==', '>', '<', '>=', '<=', '!=']
        (value) takes in an int, float, or str for the value to compare against
        '''

        # Check if dataframe has data and the specified column exists
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
                # Check for invalid condition
                self.logger.error(f"Invalid condition: {condition}")
                return None
            #
        
            # Check if any rows match the query condition
            if not result.empty:
                print(f"Query results for {query_column} {condition} {value}:")
                print(result)
                self.logger.info(f"Query results for {query_column} {condition} {value}:")
                self.logger.info(result)  # Log the query result
                return result  # Return the matching rows
            else:
                # Notify user if no rows match the condition
                print(f"No matching results found for {query_column} {condition} {value}.")
                self.logger.warning(f"No matching results found for {query_column} {condition} {value}.")
                return None
            #
        else:
            # Notify user if the dataframe is empty
            print(f"The dataframe is empty or column '{query_column}' was not found.")
            self.logger.warning(f"The dataframe is empty or column '{query_column}' was not found.")
            return None
        #
    #

    # Show stats for the month
    def show_stats_for_month(self):
        print(f'{self.name}\'s Stats:')
        print(f"\tAverage Steps per Day: {self.avg_steps}")
        print(f"\tAverage HRV per Day: {self.avg_hrv}")
        print(f"\tFitness Score: {self.fitness_score}")
        print(f"\tFitness Departure Angle: {self.departureAngle}")
    #
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Main Self-run block
if __name__ == "__main__":
    
    pers1 = FitnessDataProcessing('adam')
    pers1.show_stats_for_month()
    # pers1.query_data(pers1.df_hrv,'dayOfMonth',)

    pers2 = FitnessDataProcessing('brian')
    pers2.show_stats_for_month()
    
    pers3 = FitnessDataProcessing('charlie')
    pers3.show_stats_for_month()

    # pers4 = FitnessDataProcessing('david')
    # pers4.show_stats_for_month()

    # pers5 = FitnessDataProcessing('eddie')
    # pers5.show_stats_for_month()
    
    #TEST Code
    # main()