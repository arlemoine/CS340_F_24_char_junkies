
#%% MODULE BEGINS
module_name = 'Departments'

'''
Version: 1.0

Description:
    Module to consolidate fitness data for individuals into their department. Generates fitness statistics for the department based off of the individuals and produces relevant graphs.

Authors:
    Adriean Lemoine
    Chris Smith

Date Created     :  11/26/2024
Date Last Updated:  11/26/2024
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports
import Logging

#other imports  
from   copy       import deepcopy as dpcpy
from   matplotlib import pyplot as plt
import math
import numpy  as np 
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
from Individuals import FitnessDataProcessing
from tabulate import tabulate
from itertools import permutations, combinations
import logging

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import Config

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here


#Class definitions Start Here

class DepartmentData:
    def __init__(self, departmentName, daysInMonth=31):
        self.departmentName = departmentName
        self.individuals = []
        self.daysInMonth = daysInMonth

        self.df_dept_steps = None # Steps per day average per individual for department
        self.df_dept_hrv = None # HRV average per individual for department
        self.df_dept_fitness_scores = None # Fitness score average per individual
    #

    # Add person to group       
    def addIndividual(self, individual):
        self.individuals.append(individual)
    #

    # Remove person from group                                          
    def dropIndividual(self, individual):
        self.individuals.remove(individual)
    #

    # Examine each individual's HRV table to determine the max 'dayOfMonth' for each person.
    def calcDays(self):
        
        max_days = [] 
    
        
        for individual in self.individuals:
            if individual.df_hrv is not None and not individual.df_hrv.empty:
               
                max_day_of_month = individual.df_hrv['dayOfMonth'].max()
                max_days.append(max_day_of_month)  

        if max_days:
            days_accounted_for = min(max_days) # Use the minimum from all individuals' max values to determine the number of days accounted for
        else:
            days_accounted_for = 0

        return days_accounted_for
    #

    # Generate a DataFrame for the average steps per person in the department.
    def getSteps(self):
        
        names = []
        avg_steps = []
        department_name = self.departmentName

        for individual in self.individuals:

            individual.calc_step_score()

            avg_steps_value = round(float(individual.avg_steps), 2)

            names.append(individual.name)
            avg_steps.append(avg_steps_value)

        self.df_dept_steps = pd.DataFrame({
            'Department Name': [department_name] * len(names),
            'Officers Name': [name.capitalize() for name in names],
            'Avg_Steps': avg_steps
        })

        return self.df_dept_steps
    #

    # Generate a DataFrame for the average HRV per person in the department.
    def getHRV(self):
        
        names = []
        avg_hrv = []
        department_name = self.departmentName

        for individual in self.individuals:
            
            individual.calc_hrv_score()

            avg_hrv_value = round(float(individual.avg_hrv), 2)

            names.append(individual.name)
            avg_hrv.append(avg_hrv_value)

        self.df_dept_hrv = pd.DataFrame({
            'Department Name': [department_name] * len(names),
            'Officers Name': [name.capitalize() for name in names],
            'Avg_HRV': avg_hrv
        })

        return self.df_dept_hrv
    #

    # Generates a dataframe that shows the average fitness score for the department
    def getFitnessScores(self):
        
        names = []
        fitness_scores = []  
        department_name = self.departmentName 

        probation_score = Config.PROB_FITNESS_SCORE
        min_fitness_score = Config.MIN_FITNESS_SCORE 

        for individual in self.individuals:
            
            individual.calc_fitness_score()

            fitness_score_value = round(float(individual.fitness_score), 2)
        
            names.append(individual.name)
            fitness_scores.append(fitness_score_value)
    
        self.df_dept_fitness_scores = pd.DataFrame({
            'Department Name': [department_name] * len(names),
            'Officers Name': [name.capitalize() for name in names],
            'Avg_FitnessScore': fitness_scores
        })
        # Categorizing each individual as 'Pass', 'Probation', or 'Fail'.
        self.df_dept_fitness_scores['Status'] = self.df_dept_fitness_scores['Avg_FitnessScore'].apply(
            lambda x: 'Pass' if x >= probation_score else ('Probation' if x >= min_fitness_score else 'Fail')
        )

        return self.df_dept_fitness_scores
    #

    # Creates a dataframe that shows the average age of the department
    def getAge(self):
        names = []
        ages = []
        age_status = []  # To store the status of each individual

        for individual in self.individuals:
            names.append(individual.name)
            ages.append(individual.age)

            # Check if the individual is above or below the age of 35
            if individual.age > 35:
                age_status.append("Above 35")
            else:
                age_status.append("Below 35")

        # Create a DataFrame with the data
        df_dept_age = pd.DataFrame({
            'Department Name': [self.departmentName] * len(names),
            'Officer Name': [name.capitalize() for name in names],
            'Age': ages,
            'Age Status': age_status
        })

        return df_dept_age
    #
#

# Child class to process and save data for department variables
class DepartmentDataProcessing(DepartmentData):

    

    # Generated graphs for each variable of interest
    histSteps = None
    histHRV = None
    histFitnessScore = None
    scatAgeToFitnessScore = None

    def __init__(self, departmentName, daysInMonth, individuals):
        super().__init__(departmentName, daysInMonth)
        self.dataFile = f'{departmentName}.pkl' # Create filename for pickle file
        self.individuals = individuals
        
        self.config = {
            "DIR_OUTPUT": Config.DIR_OUTPUT
        }

        # Dataframes
        self.df_avg_steps = None
        self.df_avg_hrv = None
        self.df_age = None

        # Stats from dataframes
        self.df_stats_steps = None
        self.df_stats_hrv = None
        self.df_stats_fitness_score = None
        self.df_avg_age = None
        df_avg_fitness_score = None

        # Attributes for plot objects
        self.hist_avg_steps = None
        self.hist_avg_hrv = None
        self.hist_fitness_score = None
        self.hist_age = None
        self.scat_steps = None
    #

    # Update directory structure to Output/<department>/<individual> for each individual in group
    def updateDirectory(self):
        path = f"{self.config['DIR_OUTPUT']}/{self.departmentName}"

        if not os.path.exists(path):
            os.makedirs(path)
        #

        for person in self.individuals:
            path = f"{self.config['DIR_OUTPUT']}/{self.departmentName}/{person}"
            if not os.path.exists(path):
                os.makedirs(path)
            #
        #
    #
   
 # Generates a DataFrame that shows the minimum, mean, median, mode, and maximum of steps for the department.
    def genDf_stats_steps(self):
       
        if self.df_dept_steps is None or self.df_dept_steps.empty:
          
            self.getSteps()

        if self.df_dept_steps.empty:
            return None

        min_steps = round(self.df_dept_steps['Avg_Steps'].min(), 2)
        mean_steps = round(self.df_dept_steps['Avg_Steps'].mean(), 2)
        median_steps = round(self.df_dept_steps['Avg_Steps'].median(), 2)
        mode_steps = self.df_dept_steps['Avg_Steps'].mode()
        mode_steps = round(mode_steps[0], 2) if not mode_steps.empty else np.nan
        max_steps = round(self.df_dept_steps['Avg_Steps'].max(), 2)

        stats_df = pd.DataFrame({
             'Department Name': [self.departmentName] * 5,
            'Statistic': ['Min', 'Mean', 'Median', 'Mode', 'Max'],
            'Steps': [min_steps, mean_steps, median_steps, mode_steps, max_steps]
        })

        return stats_df
    #

    # Generates a DataFrame that shows the minimum, mean, median, mode, and maximum of HRV for the department.
    def genDf_stats_hrv(self):

        if self.df_dept_hrv is None or self.df_dept_hrv.empty:
            
            self.getHRV()
    
        if self.df_dept_hrv.empty:
            return None
    
        min_hrv = round(self.df_dept_hrv['Avg_HRV'].min(),2)
        mean_hrv = round(self.df_dept_hrv['Avg_HRV'].mean(), 2)
        median_hrv = round(self.df_dept_hrv['Avg_HRV'].median(), 2)
        mode_hrv = round(self.df_dept_hrv['Avg_HRV'].mode(), 2)
        mode_hrv = mode_hrv[0] if not mode_hrv.empty else np.nan
        max_hrv = round(self.df_dept_hrv['Avg_HRV'].max(),2)

        stats_df_hrv = pd.DataFrame({
            'Department Name': [self.departmentName] * 5,
            'Statistic': ['Min', 'Mean', 'Median', 'Mode', 'Max'],
            'HRV': [min_hrv, mean_hrv, median_hrv, mode_hrv, max_hrv]
        })

        return stats_df_hrv
    #

    # Generates a DataFrame that shows the minimum, mean, median, mode, and maximum of fitness scores for the department.
    def genDf_stats_fitness_score(self):

        if self.df_dept_fitness_scores is None or self.df_dept_fitness_scores.empty:
           
            self.getFitnessScores() 

        if self.df_dept_fitness_scores.empty:
            return None

        min_fitness_score = round(self.df_dept_fitness_scores['Avg_FitnessScore'].min(), 2)
        mean_fitness_score = round(self.df_dept_fitness_scores['Avg_FitnessScore'].mean(), 2)
        median_fitness_score = round(self.df_dept_fitness_scores['Avg_FitnessScore'].median(), 2)
        mode_fitness_score = round(self.df_dept_fitness_scores['Avg_FitnessScore'].mode(),2)
        mode_fitness_score = mode_fitness_score[0] if not mode_fitness_score.empty else np.nan
        max_fitness_score = round(self.df_dept_fitness_scores['Avg_FitnessScore'].max(),2)

        stats_df = pd.DataFrame({
            'Department Name': [self.departmentName] * 5,
            'Statistic': ['Min', 'Mean', 'Median', 'Mode', 'Max'],
            'Fitness_Score': [min_fitness_score, mean_fitness_score, median_fitness_score, mode_fitness_score, max_fitness_score]
        })

        return stats_df
    #

    # def loadData(self): # Load from a pickle file
    #     with open(fileName, 'rb') as data:
    #         temp = pkl.load(data)
    #     #
    #     return temp
    # #

    # def saveData(self): # Save to a pickle file
    #     with open(fileName, 'wb') as data:
    #         pkl.dump(self, data)
    #     #
    # #

    # def calcStatsSteps(self): # Calculate statistical values such as mean, min, max
    #     self.dfDeptStepsStats = self.deptSteps.describe()
    # #

    # def calcStatsHRV(self): # Calculate statistical values such as mean, min, max
    #     self.dfDeptHRVStats = self.deptHRV.describe()
    # #

    # def calcStatsFitnessScore(self): # Calculate statistical values such as mean, min, max
    #     self.dfDeptFitnessScoreStats = self.deptFitnessScore.describe()
    # #

    # def calcStatsAge(self): # Calculate statistical values such as mean, min, max
    #     self.dfDeptAgeStats = self.deptAge.describe()
    # #

    # def plotHistSteps(self): # Generate plot
    #     this.histSteps = self.deptSteps.plot.hist()
    # #

    # def plotHistHRV(self): # Generate plot
    #     this.histHRV = self.deptHRV.plot.hist()
    # #

    # def plotHistFitnessScore(self): # Generate plot
    #     this.histFitnessScore = self.deptFitnessScore.plot.hist()
    # #

    # def plotHistAge(self): # Generate plot
    #     self.dfAgeToFitnessScore = pd.merge(dfDeptAge, dfDeptFitnessScore, on='name')
    #     this.scatAgeToFitnessScore = self.dfAgeToFitnessScore.plot.scatter(age,fitnessScore)
    # #

    # Saves the departments data to a pickle file
    def pickleSave(self):
        # DEfines the pickle file  name based on the department name
        pickle_filename = f"{self.departmentName}.pickle"
        # Constructs the output path where the pickle file will be saved
        output_path = os.path.join(self.config['DIR_OUTPUT'], self.departmentName, pickle_filename)
        # Create the directories leading to the file path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'wb') as f:
                pkl.dump(self, f)
            print(f"Department saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving department to pickle: {e}")
    #

    # Loads the departments data from pickle file
    def pickleLoad(pickle_file_path):
   
        try:
            # Open the pickle file in read-binary mode
            with open(pickle_file_path, 'rb') as f:
                # Load the data from the pickle file
                department = pkl.load(f)
        
            # Return the loaded department data
            return department

        except FileNotFoundError:
            print(f"Error: The file '{pickle_file_path}' was not found.")
        except pkl.UnpicklingError:
            print(f"Error: There was an issue unpickling the file '{pickle_file_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    #

    #
    def genVectors(self, name):
        def project_vector(v, u):
            '''
            Projects vector v onto vector u.
            v: The vector to be projected.
            u: The vector onto which v is projected.
            '''

            # Calculate the dot product of v and u
            dot_product = np.dot(v, u)
            
            # Calculate the magnitude of u squared
            magnitude_squared = np.linalg.norm(u) ** 2
            
            # Calculate the projection
            projection = (dot_product / magnitude_squared) * u
            
            return projection
        #
        
        # Define vectors
        origin = np.array([0,0])
        proj_v1 = np.array([self.df_avg_age,self.df_avg_fitness_score]) # Vector representing departmental average
        proj_v2 = np.array([self.individuals[name].age,self.individuals[name].fitness_score]) # Vector representing individual
        proj_v3 = project_vector(proj_v2, proj_v1) # Projection vector
        angle_v1 = proj_v1 / np.linalg.norm(proj_v1)
        angle_v2 = proj_v2 / np.linalg.norm(proj_v2)
        
        # Generate and configure subplots
        fig, axs = plt.subplots(1,2,figsize=(10,4))
        axs[0].set_xlim(0,5)
        axs[0].set_ylim(0,5)
        axs[1].set_xlim(0,2)
        axs[1].set_ylim(0,2)
        axs[0].set_title('Projection Vector')
        axs[1].set_title('Departure Angle')
        
        # Generate vectors for subplot 1
        axs[0].quiver(origin[0], origin[1], proj_v1[0], proj_v1[1], angles='xy', scale_units='xy', scale=1, color='b', label='Average')
        axs[0].quiver(origin[0], origin[1], proj_v2[0], proj_v2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Individual')
        axs[0].quiver(origin[0], origin[1], proj_v3[0], proj_v3[1], angles='xy', scale_units='xy', scale=1, color='r', label='Projection')
        
        # Generate vectors for subplot 2
        axs[1].quiver(origin[0], origin[1], angle_v1[0], angle_v1[1], angles='xy', scale_units='xy', scale=1, color='b', label='Average')
        axs[1].quiver(origin[0], origin[1], angle_v2[0], angle_v2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Individual')
        
        # Apply legend
        axs[0].legend()
        axs[1].legend()

        # Update individual departure angle
        self.individuals[name].departureAngle = np.arccos(np.dot(angle_v1, angle_v2)) * 180 / math.pi

        # Save figure as png
        fig.savefig(f"{self.config['DIR_OUTPUT']}/{self.departmentName}/{name}/vectors.png")    
    #

    # Writes out the stats of each individual in the department and saves their data to a file
    def writeIndividualStats(self):
        
        for individual in self.individuals:
            
            individual_dir = f"{self.config['DIR_OUTPUT']}/{self.departmentName}/{individual.name}"
        
            if not os.path.exists(individual_dir):
                os.makedirs(individual_dir)
        
            
            individual_stats = []

            # Collect stats for this individual (steps, HRV, fitness score)
            individual.calc_step_score()
            individual.calc_hrv_score()
            individual.calc_fitness_score()

            # Format the individual's stats into a list of strings
            individual_stats.append(f"Name: {individual.name.capitalize()}")
            individual_stats.append(f"Age: {individual.age}")
            individual_stats.append(f"Avg Steps: {round(individual.avg_steps, 2)}")
            individual_stats.append(f"Avg HRV: {round(individual.avg_hrv, 2)}")
            individual_stats.append(f"Fitness Score: {round(individual.fitness_score, 2)}")
            individual_stats.append(f"Status: {self.df_dept_fitness_scores[self.df_dept_fitness_scores['Officers Name'] == individual.name.capitalize()]['Status'].values[0]}")

            # Join the stats into a single string, one per line
            stats_str = "\n".join(individual_stats)

            # Write the stats to a file
            stats_file_path = f"{individual_dir}/indStats.txt"
            with open(stats_file_path, "w") as file:
                file.write(stats_str)
    #

    # Writes out the departments stats to a file
    def writeDepartmentStats(self):
        # Prepares the departments statistics
        stats = []
        stats.append(f"Department: {self.departmentName}")
        stats.append("-" * 50)

        # Averages the statistics for the department
        avg_steps = self.df_dept_steps['Avg_Steps'].mean()
        avg_hrv = self.df_dept_hrv['Avg_HRV'].mean()
        avg_fitness_score = self.df_dept_fitness_scores['Avg_FitnessScore'].mean()

        stats.append(f"1. Average Steps per Person: {round(avg_steps, 2)}")
        stats.append(f"2. Average HRV per Person: {round(avg_hrv, 2)}")
        stats.append(f"3. Average Fitness Score per Person: {round(avg_fitness_score, 2)}")

        stats.append("\nStatistics for the Department:\n")

        # Writes the statistics for steps
        steps_stats = self.genDf_stats_steps()
        stats.append("Steps Statistics:\n" + tabulate(steps_stats, headers='keys', tablefmt='plain'))

        # Writes the statistics for HRV
        hrv_stats = self.genDf_stats_hrv()
        stats.append("\nHRV Statistics:\n" + tabulate(hrv_stats, headers='keys', tablefmt='plain'))

        # Writes the statistics for fitness scores
        fitness_score_stats = self.genDf_stats_fitness_score()
        stats.append("\nFitness Score Statistics:\n" + tabulate(fitness_score_stats, headers='keys', tablefmt='plain'))

        # Writes the status for each individual
        stats.append("\nStatus Summary:")
        stats.append("-" * 50)
        for individual in self.individuals:
            status = self.df_dept_fitness_scores[self.df_dept_fitness_scores['Officers Name'] == individual.name.capitalize()]['Status'].values[0]
            stats.append(f"{individual.name.capitalize()}  | {status}")

        # Creates the file path for the department
        output_file_path = f"{self.config['DIR_OUTPUT']}/{self.departmentName}/deptStats.txt"

        # Writes to the file
        with open(output_file_path, "w") as file:
            file.write("\n".join(stats))
    #

    #
     # Method to get unique values from a categorical attribute
    def get_unique_values(self, column_name):
        """
        Extract unique values from a given categorical attribute column.
        """
        if column_name in self.df_dept_fitness_scores.columns:
            unique_values = self.df_dept_fitness_scores[column_name].unique()
            return unique_values
        else:
            print(f"Column '{column_name}' not found.")
            return None
    #

    #
     # Method to generate all permutations of unique values from a categorical attribute
    def get_permutations(self, column_name, r=None):
        """
        Generate all permutations of unique values from the given column.
        """
        unique_values = self.get_unique_values(column_name)
        if unique_values is not None:
            if r is None:
                r = len(unique_values)  # Default to full-length permutations
            return list(permutations(unique_values, r))
        else:
            return None
    #

    #
    # Method to generate all combinations of unique values from a categorical attribute
    def get_combinations(self, column_name, r=None):
        """
        Generate all combinations of unique values from the given column.
        """
        unique_values = self.get_unique_values(column_name)
        if unique_values is not None:
            if r is None:
                r = len(unique_values)  # Default to full-length combinations
            return list(combinations(unique_values, r))
        else:
            return None
    #
    #
#
if __name__ == "__main__":

# Creates the department and adds the departments name, daysInMonth, and individuals
    dept1 = DepartmentDataProcessing('CMPSPD 340', daysInMonth=30, individuals=[])
    

# Create instances of FitnessDataProcessing (each individual)
    person1 = FitnessDataProcessing('adam')
    person2 = FitnessDataProcessing('brian')
    person3 = FitnessDataProcessing('charlie')
    person4 = FitnessDataProcessing('david')
    person5 = FitnessDataProcessing('eddie')

# Add individuals to the department
    dept1.addIndividual(person1)
    dept1.addIndividual(person2)
    dept1.addIndividual(person3)
    dept1.addIndividual(person4)
    dept1.addIndividual(person5)

# Get steps data for the department (average steps per person)
    steps_df = dept1.getSteps()
    hrv_df = dept1.getHRV()
    fitnessScores_df = dept1.getFitnessScores()
    days_accounted_for = dept1.calcDays()
    
#Get the dataframe with the individual steps
    df_individual_steps = dept1.getSteps()

# Get the average age for each department
    df_dept_age = dept1.getAge()

# After adding individuals to the department and generating step data
    stats_steps_df = dept1.genDf_stats_steps()

# Get the stats for HRV
    stats_HRV_df = dept1.genDf_stats_hrv()

# Get the stats for fitness scores
    stats_fitnessScores_df = dept1.genDf_stats_fitness_score()

# Get the stats for each individual in the department
    dept1.writeIndividualStats()

# Get the stats for the entire department
    dept1.writeDepartmentStats()

# Save to pickle file
    dept1.pickleSave()

# Defining the pickle file to load
    pickle_file_path = r"C:\DevTools\CMPS 340 Project\CS340_F_24_char_junkies-2\Output\CMPSPD 340\CMPSPD 340.pickle"

# Loading the data from the pickle file
    dept1_data = DepartmentDataProcessing.pickleLoad(pickle_file_path)

# Check the loaded department data
    if dept1_data:
        print("Pickle data loaded successfully:")
        print(dept1_data) 
        print(f"Department Name: {dept1_data.departmentName}")
        print(f"Average steps: {dept1_data.df_dept_steps['Avg_Steps'].mean()}")
        
    else:
        print("Failed to load pickle data.")
    
# To get unique values from the "Status" column:
    unique_status_values = dept1.get_unique_values("Status")
    print(f"Unique Status Values: {unique_status_values}")



# combinations and permutations of a smaller
    status_combinations = dept1.get_combinations("Status")
    status_permutations = dept1.get_permutations("Status")

    print(f"Status Combinations: {status_combinations}")
    print(f"Status Permutations: {status_permutations}")

    


# Display the department's average steps per individual
    print(f"Days accounted for: {days_accounted_for}")
    print(tabulate(steps_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(hrv_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(fitnessScores_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(df_dept_age, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(stats_steps_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(stats_HRV_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(stats_fitnessScores_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    
    
