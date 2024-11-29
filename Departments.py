
#%% MODULE BEGINS
module_name = 'Departments'

'''
Version: 1.0

Description:
    Module to consolidate fitness data for individuals into their department. Generates fitness statistics for the department based off of the individuals and produces relevant graphs.

Authors:
    Adriean Lemoine

Date Created     :  11/26/2024
Date Last Updated:  11/26/2024
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports

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

    #
    def getSteps(self):
        """
        Generate a DataFrame for the average steps per person in the department.
        It calculates the average steps for each person using `calc_step_score()`.
        """
        # Start logging method entry
        # self.logger.info("Starting to generate the department's average steps data.")

        names = []
        avg_steps = []
        department_name = self.departmentName

        # Log the department being processed
        # self.logger.info(f"Processing department: {department_name}")

        # Loop through each individual in the department
        for individual in self.individuals:
            # self.logger.info(f"Calculating step score for individual: {individual.name}")
            individual.calc_step_score()  # Calculate the step score (average steps per day)

            # Capture the result of avg_steps calculation
            avg_steps_value = round(individual.avg_steps, 2)
            # self.logger.debug(f"{individual.name}'s average steps: {avg_steps_value}")

            # Append the results to the lists
            names.append(individual.name)
            avg_steps.append(avg_steps_value)

        # Create DataFrame
        self.df_dept_steps = pd.DataFrame({
            'Department Name': [department_name] * len(names),
            'Officers Name': [name.capitalize() for name in names],
            'Avg_Steps': avg_steps
        })

        # Log DataFrame creation
        # self.logger.info(f"Generated DataFrame with {len(self.df_dept_steps)} rows for department: {department_name}")

        # Return the DataFrame
        return self.df_dept_steps
    #

    #
    def getHRV(self):
        """
        Generate a DataFrame for the average HRV per person in the department.
        It calculates the average HRV for each person using their `df_hrv` DataFrame.
        """
        # Start logging method entry
        # self.logger.info("Starting to generate the department's average HRV data.")

        names = []
        avg_hrv = []
        department_name = self.departmentName

        # Log the department being processed
        # self.logger.info(f"Processing department: {department_name}")

        # Loop through each individual in the department
        for individual in self.individuals:
            # self.logger.info(f"Calculating average HRV for individual: {individual.name}")

            # Calculate the average HRV directly
            avg_hrv_value = individual.df_hrv['hrv'].mean()  # Calculate the average HRV
            avg_hrv_value = round(avg_hrv_value, 2)

            # Log the calculated average HRV for this individual
            # self.logger.debug(f"{individual.name}'s average HRV: {avg_hrv_value}")

            # Append the results to the lists
            names.append(individual.name)
            avg_hrv.append(avg_hrv_value)

        # Create DataFrame
        self.df_dept_hrv = pd.DataFrame({
            'Department Name': [department_name] * len(names),
            'Officers Name': [name.capitalize() for name in names],
            'Avg_HRV': avg_hrv
        })

        # Log DataFrame creation
        # self.logger.info(f"Generated DataFrame with {len(self.df_dept_hrv)} rows for department: {department_name}")

        # Return the DataFrame
        return self.df_dept_hrv
    #

    #
    def getFitnessScores(self):
        """
        Generate a DataFrame for the average fitness score per person in the department.
        It calculates the fitness score for each person using `calc_fitness_score()`.
        """
        # Start logging method entry
        # self.logger.info("Starting to generate the department's fitness scores data.")

        names = []
        fitness_scores = []  
        department_name = self.departmentName 

        # Log the department being processed
        # self.logger.info(f"Processing department: {department_name}")

        # Loop through each individual in the department
        for individual in self.individuals:
            # self.logger.info(f"Calculating fitness score for individual: {individual.name}")
        
            # Call the method to calculate the fitness score
            individual.calc_fitness_score()  # Calculates the fitness score for the individual

            # Capture the result of avg_FitnessScore calculation
            fitness_score_value = round(individual.fitness_score, 2)  # Round to 2 decimal places
            # self.logger.debug(f"{individual.name}'s fitness score: {fitness_score_value}")
            
            names.append(individual.name)
            fitness_scores.append(fitness_score_value)
            
        # Create DataFrame for the department's fitness scores
        self.df_dept_fitness_scores = pd.DataFrame({
            'Department Name': [department_name] * len(names),
            'Officers Name': [name.capitalize() for name in names],
            'Avg_FitnessScore': fitness_scores
        })

        # Log DataFrame creation
        # self.logger.info(f"Generated DataFrame with {len(self.df_dept_fitness_scores)} rows for department: {department_name}")

        # Return the DataFrame
        return self.df_dept_fitness_scores
    #

    #
    def calcDays(self):
        """
        Examine each individual's HRV table to determine the max 'dayOfMonth' for each person.
        Then, use the minimum from all individuals' max values to determine the number of days accounted for.
        """
        max_days = []  # List to store the max 'dayOfMonth' for each individual
    
        # Loop through each individual in the department
        for individual in self.individuals:
            if individual.df_hrv is not None and not individual.df_hrv.empty:
                # Extract the max 'dayOfMonth' value from the HRV DataFrame
                max_day_of_month = individual.df_hrv['dayOfMonth'].max()
                max_days.append(max_day_of_month)  # Append to the list of max days

        # Determine the minimum max 'dayOfMonth' value from all individuals
        if max_days:
            days_accounted_for = min(max_days)
        else:
            days_accounted_for = 0  # If there are no days, return 0

        # Return the minimum of the max days from all individuals
        return days_accounted_for
    #
#

# Child class to process and save data for department variables
class DepartmentDataProcessing(DepartmentData):

    

    # Generated graphs for each variable of interest
    histSteps = None
    histHRV = None
    histFitnessScore = None
    scatAgeToFitnessScore = None

    def __init__(self, departmentName, daysInMonth):
        super().__init__(departmentName, daysInMonth)
        self.dataFile = f'{departmentName}.pkl' # Create filename for pickle file
        
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
#
if __name__ == "__main__":
    # Create department and add individuals
    dept1 = DepartmentData('CMPSPD 340')

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



# Display the department's average steps per individual
    print(tabulate(steps_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(hrv_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(tabulate(fitnessScores_df, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
    print(f"Days accounted for: {days_accounted_for}")
