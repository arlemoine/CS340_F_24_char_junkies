#%% MODULE BEGINS
module_name = 'Departments'

'''
Version: 1.0.0

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
import mne
import numpy  as np 
import os
import pandas as pd
import pickle as pkl
import seaborn as sns

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import Config

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here


#Class definitions Start Here

class DepartmentData:
    def __init__(self, departmentName, daysInMonth):
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

    # def getSteps(self): # Generate dataframe for average steps per person
    #     names = []
    #     avgSteps = []

    #     for individual in individuals:
    #         names.append(individual.name)
    #         avgSteps.append(individual.avgSteps)

    #     self.dfDeptSteps = pd.DataFrame({
    #         'Name': names,
    #         'avgSteps': avgSteps
    #     })
    # #

    # def getHRV(self): # Generate dataframe for average HRV per person
    #     names = []
    #     avgHRV = []

    #     for individual in individuals:
    #         names.append(individual.name)
    #         avgHRV.append(individual.avgHRV)

    #     self.dfDeptHRV = pd.DataFrame({
    #         'Name': names,
    #         'avgHRV': avgHRV
    #     })
    # #

    # def getFitnessScores(self): # Generate dataframe for average HRV per person
    #     names = []
    #     avgFitnessScore = []

    #     for individual in individuals:
    #         names.append(individual.name)
    #         avgFitnessScore.append(individual.avgFitnessScore)

    #     self.dfDeptFitnessScore = pd.DataFrame({
    #         'Name': names,
    #         'avgFitnessScore': avgFitnessScore
    #     })
    # #
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
        path = f'{self.config['DIR_OUTPUT']}/{self.departmentName}'

        if not os.path.exists(path):
            os.makedirs(path)
        #

        for person in self.individuals:
            path = f'{self.config['DIR_OUTPUT']}/{self.departmentName}/{person}'
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
        fig.savefig(f'{self.config['DIR_OUTPUT']}/{self.departmentName}/{name}/vectors.png')    
    #
#
