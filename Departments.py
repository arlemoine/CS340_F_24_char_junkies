#%% MODULE BEGINS
module_name = 'Departments'

'''
Version: 1.3

Description:
    Module to consolidate fitness data for individuals into their department. Generates fitness statistics for the department based off of the individuals and produces relevant graphs.

Authors:
    Adriean Lemoine
    Chris Smith
    Nicholas Burgo

Date Created     :  11/26/2024
Date Last Updated:  12/05/2024
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports
from Individuals import FitnessDataProcessing
import Logging

#other imports  
import contextlib as clib # Used to redirect output stream from terminal to a file for saving individual info
from   copy       import deepcopy as dpcpy
from itertools import combinations, permutations
import math
from   matplotlib import pyplot as plt
import numpy  as np 
import os
import pandas as pd
import pickle as pkl
from tabulate import tabulate

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import Config

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here


#Class definitions Start Here

class DepartmentData:
    individuals = {}
    df_dept_steps = None # Steps per day average per individual for department
    df_dept_hrv = None # HRV average per individual for department
    df_dept_fitness_scores = None # Fitness score average per individual
    df_dept_age = None # Age per individual

    def __init__(self, departmentName, daysInMonth=31):
        self.departmentName = departmentName
        self.daysInMonth = daysInMonth

        self.initLog()
    #

    def initLog(self):
        self.logger = Logging.configure_logger(self.departmentName, f'Output/{self.departmentName}/')
    #

    # Add person to group       
    def addIndividual(self, individual):
        self.individuals[individual.name] = individual
        self.logger.info(f"{individual.name} added to department.")
    #

    # Remove person from group                                          
    def dropIndividual(self, individual):
        if individual.name in self.individuals:
            del self.individuals[individual.name]
            self.logger.info(f"{individual.name} removed from department.")
        else:
            print(f'Individual named {individual.name} not found in department.')
    #

    # Examine each individual's HRV table to determine the max 'dayOfMonth' for each person.
    def calcDays(self):
        
        max_days = [] 
    
        
        for individual in self.individuals.values():
            if individual.df_hrv is not None and not individual.df_hrv.empty:
               
                max_day_of_month = individual.df_hrv['dayOfMonth'].max()
                max_days.append(max_day_of_month)  

        if max_days:
            days_accounted_for = min(max_days) # Use the minimum from all individuals' max values to determine the number of days accounted for
        else:
            days_accounted_for = 0

        return days_accounted_for
    #

    # Create all dataframes
    def getDataframes(self):
        self.df_dept_steps = self.getSteps()
        self.df_dept_hrv = self.getHRV()
        self.df_dept_fitness_scores = self.getFitnessScores()
        self.df_dept_age = self.getAges()
    #

    # Generate a DataFrame for the average steps per person in the department.
    def getSteps(self):
        names = []
        avgSteps = []
        department_name = self.departmentName

        for individual in self.individuals.values():
            # individual.calc_step_score()
            # avgSteps_value = round(float(individual.avg_steps), 2)
            names.append(individual.name)
            avgSteps.append(individual.avg_steps)
        #

        self.df_dept_steps = pd.DataFrame({
            'deptName': [department_name] * len(names),
            'indName': [name.capitalize() for name in names],
            'avgSteps': avgSteps
        })

        self.logger.info(f"Steps dataframe retrieved.")
        return self.df_dept_steps
    #

    # Generate a DataFrame for the average HRV per person in the department.
    def getHRV(self):
        names = []
        avgHRV = []
        department_name = self.departmentName

        for individual in self.individuals.values():
            # individual.calc_hrv_score()
            # avgHRV_value = round(float(individual.avg_hrv), 2)
            names.append(individual.name)
            avgHRV.append(individual.avg_hrv)
        #

        self.df_dept_hrv = pd.DataFrame({
            'deptName': [department_name] * len(names),
            'indName': [name.capitalize() for name in names],
            'avgHRV': avgHRV
        })

        self.logger.info(f"HRV dataframe retrieved.")
        return self.df_dept_hrv
    #

    # Generates a dataframe that shows the average fitness score for the department
    def getFitnessScores(self):
        names = []
        fitness_scores = []  
        department_name = self.departmentName 
        probation_score = Config.PROB_FITNESS_SCORE
        min_fitness_score = Config.MIN_FITNESS_SCORE 

        for individual in self.individuals.values():
            # individual.calc_fitness_score()
            # fitness_score_value = round(float(individual.fitness_score), 2)
            names.append(individual.name)
            fitness_scores.append(individual.fitness_score)
        #
    
        self.df_dept_fitness_scores = pd.DataFrame({
            'deptName': [department_name] * len(names),
            'indName': [name.capitalize() for name in names],
            'fitnessScore': fitness_scores
        })

        # Categorizing each individual as 'Pass', 'Probation', or 'Fail'.
        self.df_dept_fitness_scores['fitnessGroup'] = self.df_dept_fitness_scores['fitnessScore'].apply(
            lambda x: 'Pass' if x >= probation_score else ('Probation' if x >= min_fitness_score else 'Fail')
        )

        self.logger.info(f"Fitness score data retrieved.")
        return self.df_dept_fitness_scores
    #

    # Creates a dataframe that shows the average age of the department
    def getAges(self):
        names = []
        ages = []
        age_status = []  # To store the status of each individual

        for individual in self.individuals.values():
            names.append(individual.name)
            ages.append(individual.age)

            # Check if the individual is above or below the age of 35
            if individual.age > 35:
                age_status.append("Above 35")
            else:
                age_status.append("Below 35")

        # Create a DataFrame with the data
        df_dept_age = pd.DataFrame({
            'deptName': [self.departmentName] * len(names),
            'indName': [name.capitalize() for name in names],
            'age': ages,
            'ageGroup': age_status
        })

        self.logger.info(f"Personnel age information retrieved.")
        return df_dept_age
    #
#

# Child class to process and save data for department variables
class DepartmentDataProcessing(DepartmentData):
    config = {
        "DIR_OUTPUT": Config.DIR_OUTPUT
    }
    
    # Stats from dataframes
    df_stats_steps = None
    df_stats_hrv = None
    df_stats_fitness_score = None
    df_stats_age = None
    df_jointCounts = None
    df_jointProbs = None

    # Statistics variables
    totalCounts = None
    condProbs = {}
    uniqueAgeGroup = None
    uniqueFitnessGroup = None
    combAgeGroup = None
    combFitnessGroup = None
    permAgeGroup = None
    permFitnessGroup = None

    def __init__(self, departmentName, daysInMonth):
        super().__init__(departmentName, daysInMonth)
        self.__dirPickle = f"Output/{departmentName}/{departmentName}.pkl" # Create filename for pickle file
    #

   # Write all data to proper files
    def getAll(self):
        self.updateDirectory()
        self.getDataframes()
        self.logger.info('Data retrieved for individuals.')
        self.calcDays()
        self.getStats_all()
        self.logger.info('Statistics generated from data.')
        self.gen_dept_hist()
        self.logger.info('Histograms created.')
        self.gen_dept_vectors()
        self.logger.info('Vector graphs created for individuals.')
        self.writeStats()
        self.logger.info('Department data written to output folder.')
        self.pickleSave()
        self.logger.info('Department saved to pickle file.')
    #

    # Create folder for dept in Output folder
    def updateDirectory(self):
        path = f"{self.config['DIR_OUTPUT']}/{self.departmentName}"

        if not os.path.exists(path):
            os.makedirs(path)
        #
    #

    # Perform methods necessary to gather statistical data
    def getStats_all(self):
        self.getStats_steps()
        self.getStats_hrv()
        self.getStats_fitness_score()
        self.getStats_age()
        self.getJointCounts()
        self.getJointProbs()
        self.getCondProbs()
        self.getUniqueValues()
    #

    # Generates dataframe for step statistics
    def getStats_steps(self):
        if self.df_dept_steps is None or self.df_dept_steps.empty:
            self.getSteps()
            
        #
        if self.df_dept_steps.empty:
            return None
        #

        self.df_stats_steps = self.df_dept_steps.describe().round(2)
        self.logger.info(f"Steps statistics calculated.")
    #

    # Generates dataframe for HRV statistics
    def getStats_hrv(self):
        if self.df_dept_hrv is None or self.df_dept_hrv.empty:
            self.getHRV()
        #
        if self.df_dept_hrv.empty:
            return None
        #

        self.df_stats_hrv = self.df_dept_hrv.describe().round(2)
        self.logger.info(f"HRV statistics calculated.")
    #

    # Generates dataframe for fitness score statistics
    def getStats_fitness_score(self):
        if self.df_dept_fitness_scores is None or self.df_dept_fitness_scores.empty:
            self.getFitnessScores() 
        #
        if self.df_dept_fitness_scores.empty:
            return None
        #

        self.df_stats_fitness_score = self.df_dept_fitness_scores.describe().round(2)
        self.logger.info(f"Fitness score statistics calculated.")
    #

    # Generates dataframe for age statistics
    def getStats_age(self):
        if self.df_dept_age is None or self.df_dept_age.empty:
            self.getAges() 
        #
        self.df_stats_age = self.df_dept_age.describe().round(2)
        self.logger.info(f"Age statistics calculated.")
    #

    # Generate subplots for histograms of data
    def gen_dept_hist(self):
        graph_dir = f"{self.config['DIR_OUTPUT']}{self.departmentName}/"

        # Prepare data for histograms
        if self.df_dept_steps is None:
            self.getSteps()
        if self.df_dept_hrv is None:
            self.getHRV()
        if self.df_dept_fitness_scores is None:
            self.getFitnessScores()
        avgSteps = self.df_dept_steps['avgSteps']
        avgHRV = self.df_dept_hrv['avgHRV']
        fitness_scores = self.df_dept_fitness_scores['fitnessScore']
        ages = [indiv.age for indiv in self.individuals.values()]

        # Create a single figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Histograms for Department: {self.departmentName}", fontsize=16)

        # Histogram for Average Steps
        axs[0, 0].hist(avgSteps, bins=5, alpha=0.7, edgecolor='black')
        axs[0, 0].set_title("Average Steps per Individual")
        axs[0, 0].set_xlabel("Average Steps")
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

        # Histogram for Average HRV
        axs[0, 1].hist(avgHRV, bins=4, alpha=0.7, edgecolor='black')
        axs[0, 1].set_title("Average HRV per Individual")
        axs[0, 1].set_xlabel("Average HRV")
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

        # Histogram for Fitness Score
        axs[1, 0].hist(fitness_scores, bins=5, alpha=0.7, edgecolor='black')
        axs[1, 0].set_title("Fitness Score per Individual")
        axs[1, 0].set_xlabel("Fitness Score")
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

        # Histogram for Age
        axs[1, 1].hist(ages, bins=5, alpha=0.7, edgecolor='black')
        axs[1, 1].set_title("Age per Individual")
        axs[1, 1].set_xlabel("Age")
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(graph_dir, "histograms.png")
        plt.savefig(output_path)
        plt.close()

        self.logger.info        (f"Combined histograms saved at: {output_path}")
    #

    # Generate dataframe for joint counts related to age and fitness score categories
    def getJointCounts(self):
        self.df_jointCounts = pd.crosstab(self.df_dept_age['ageGroup'], self.df_dept_fitness_scores['fitnessGroup'], margins=True)
        self.totalCounts = self.df_jointCounts.loc['All','All']
        self.logger.info("Recorded joint counts.")
    #

    # Generate dataframe for joint probabilities related to age and fitness score categories
    def getJointProbs(self):
        self.df_jointProbs = self.df_jointCounts / self.totalCounts
        self.getCondProbs()
        self.logger.info("Recorded joint probabilities.")
    #

    # Fills dictionary 'condProbs' with P(B|A) values
    def getCondProbs(self):
        # Iterates through rows for conditional probabilities
        for event in self.df_jointProbs.index:
            # Iterates through columns for conditional probabilities
            for outcome in self.df_jointProbs.columns:
                # Ensures that margin column and row are not used improperly
                if event != 'All' and outcome != 'All':
                    condProb1 = self.df_jointProbs.loc[event, outcome] / self.df_jointProbs.loc['All', outcome]
                    condProb2 = self.df_jointProbs.loc[event, outcome] / self.df_jointProbs.loc[event, 'All']
                    self.condProbs[f"P({event}|{outcome})"] = condProb1
                    self.condProbs[f"P({outcome}|{event})"] = condProb2
                #
            #
        #
    #

    def getUniqueValues(self):
        self.uniqueAgeGroup = self.df_dept_age['ageGroup'].unique()
        self.uniqueFitnessGroup = self.df_dept_fitness_scores['fitnessGroup'].unique()
        self.combAgeGroup = combinations(self.uniqueAgeGroup, 1) # Ways to choose 1 out of 2 age groups
        self.combFitnessGroup = combinations(self.uniqueFitnessGroup, 2) # Ways to choose 2 out of 3 fitness groups
        self.permAgeGroup = permutations(self.uniqueAgeGroup, 2)
        self.permFitnessGroup = permutations(self.uniqueFitnessGroup, 3)
        self.logger.info(f"Recorded informnation for age/fitness group combinations and permutations.")
    #

    # Create vector plots for all personnel in department
    def gen_dept_vectors(self):
        avgAge = self.df_stats_age.loc['mean','age']
        avgFitnessScore = self.df_stats_fitness_score.loc['mean','fitnessScore']

        for person in self.individuals.values():
            self.gen_vectors(person, avgAge, avgFitnessScore)
        #

        self.logger.info(f"Vector diagrams generated for all personnel.")
    #

    # Create vector plots for individual
    def gen_vectors(self, person, avgAge, avgFitnessScore):
        def project_vector():
            '''
            Projects v onto u.
            '''
            nonlocal v
            nonlocal u

            return (np.dot(v,u)/np.dot(u,u)) * u
        #
        
        # Normalization required to see orthogonal relationship between projection and v 
        normalize = lambda x, min, max: (x - min) / (max - min)
        proj_persAge = normalize(person.age,18,65)
        proj_persFit = normalize(person.fitness_score,10,100)
        proj_deptAge = normalize(avgAge,18,65)
        proj_deptFit = normalize(avgFitnessScore,10,100)

        # Define vectors for subplot 1
        origin = np.array([0,0])
        v = np.array([proj_persAge,proj_persFit]) # Individual
        u = np.array([proj_deptAge,proj_deptFit]) # Average
        p = project_vector() # Projection vector

        # Define vectors for subplot 2
        indRaw = np.array([person.age,person.fitness_score])
        deptRaw = np.array([avgAge,avgFitnessScore])
        indNorm = indRaw / np.linalg.norm(indRaw)
        deptNorm = deptRaw / np.linalg.norm(deptRaw)
        
        # Generate and configure subplots
        fig, axs = plt.subplots(1,2,figsize=(10,4))
        axs[0].set_xlim(0,1)
        axs[0].set_ylim(0,1)
        axs[1].set_xlim(0,1)
        axs[1].set_ylim(0,1)
        axs[0].set_title('Projection Vector')
        axs[1].set_title('Departure Angle')
        
        # Generate vectors for subplot 1
        axs[0].quiver(origin[0], origin[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Individual')
        axs[0].quiver(origin[0], origin[1], u[0], u[1], angles='xy', scale_units='xy', scale=1, color='g', label='Average')
        axs[0].quiver(origin[0], origin[1], p[0], p[1], angles='xy', scale_units='xy', scale=1, color='r', label='Projection')
        axs[0].quiver(p[0], p[1], v[0] - p[0], v[1] - p[1], angles='xy', scale_units='xy', scale=1, color='y', label='Orthogonality') # Visual representation of orthogonality
        
        # Generate vectors for subplot 2
        axs[1].quiver(origin[0], origin[1], indNorm[0], indNorm[1], angles='xy', scale_units='xy', scale=1, color='b', label='Average')
        axs[1].quiver(origin[0], origin[1], deptNorm[0], deptNorm[1], angles='xy', scale_units='xy', scale=1, color='g', label='Individual')
        
        # Apply legend
        axs[0].legend()
        axs[1].legend()

        # Update individual departure angle
        person.departureAngle = np.arccos(np.dot(indNorm, deptNorm)) * 180 / math.pi
        person.writeAllToFile()

        # Save figure as png
        fig.savefig(f"{self.config['DIR_OUTPUT']}/{person.name}/vectors.png")
        plt.close()
        person.logger.info('Vector graph saved to output folder.')
    #

    # Writes out the departments stats to a file
    def writeStats(self, **kwargs): 
        defaultDir = f'Output/{self.departmentName}/data.txt'
        dir = kwargs.get('filename', defaultDir)

        with open(dir, "w") as f:
            with clib.redirect_stdout(f):
                print(f'Department Name:\t{self.departmentName}')
                print(f'People:')
                for person in self.individuals.values():
                    print(f'\t{person.name}')
                #

                print(f'\nSTATISTICS\n===========')
                print(f'\n{self.df_stats_steps}')
                print(f'\n{self.df_stats_hrv}')
                print(f'\n{self.df_stats_fitness_score}')
                print(f'\n{self.df_stats_age}')
                print(f"\nJoint Counts:")
                print(f'{self.df_jointCounts}')
                print(f"\nJoint Probabilities:")
                print(f'{self.df_jointProbs}')
                
                print(f"\nConditional Probabilities:")
                for key, value in self.condProbs.items():
                    print(f"\t{key} = {value}")
                #

                print(f"\nAge Group Combinations, 2 choose 1")
                for i in list(self.combAgeGroup):
                    print(f"\t{i}")
                #
                print(f"\nAge Group Permutations")
                for i in list(self.permAgeGroup):
                    print(f"\t{i}")
                #
                print(f"\nFitness Group Combinations, 2 choose 1")
                for i in list(self.combFitnessGroup):
                    print(f"\t{i}")
                #
                print(f"\nFitness Group Permutations")
                for i in list(self.permFitnessGroup):
                    print(f"\t{i}")
                #

                print(f'\nDATA\n=====')
                print(f'\n{self.df_dept_steps}')
                print(f'\n{self.df_dept_hrv}')
                print(f'\n{self.df_dept_fitness_scores}')
                print(f'\n{self.df_dept_age}')
            #
        #

        self.logger.info(f"Department data written to {dir}.")
    #

    # Saves the departments data to a pickle file
    def pickleSave(self):
        try:
            with open(self.__dirPickle, 'wb') as f:
                pkl.dump(self, f)
            self.logger.info(f"Department saved successfully to {self.__dirPickle}")
        except Exception as e:
            print(f"Error saving department to pickle: {e}")
            self.logger.info(f"Error saving department to pickle: {e}")
        #
    #

    # Loads the departments data from pickle file
    @classmethod
    def pickleLoad(cls, filePath):
        try:
            # Open the pickle file in read-binary mode
            with open(filePath, 'rb') as f:
                obj = pkl.load(f)

            for person in obj.individuals.values():
                person.writeOutputs()

            obj.writeOutputs()
            # Return the loaded department data
            return obj
        except FileNotFoundError:
            print(f"Error: The file '{filePath}' was not found.")
        except pkl.UnpicklingError:
            print(f"Error: There was an issue unpickling the file '{filePath}'.")
        # except Exception as e:
        #     print(f"An error occurred: {e}")
    #

   # Used following a pickle load to regenerate output files
    def writeOutputs(self):
        self.updateDirectory()
        self.initLog()
        self.gen_dept_hist()
        self.gen_dept_vectors()
        self.writeStats()
        self.pickleSave()
    #

    # Print list of individuals in department
    def printIndividuals(self):
        for person in self.individuals.values():
            person.print('name','age','fitness_score')
            print()
        #
    #
#

if __name__ == "__main__":

    # Generate test department
    dept1 = DepartmentDataProcessing('CMPSPD 340', daysInMonth=30)
    person1 = FitnessDataProcessing('adam')
    person2 = FitnessDataProcessing('brian')
    person3 = FitnessDataProcessing('charlie')
    person4 = FitnessDataProcessing('david')
    person5 = FitnessDataProcessing('eddie')
    dept1.addIndividual(person1)
    dept1.addIndividual(person2)
    dept1.addIndividual(person3)
    dept1.addIndividual(person4)
    dept1.addIndividual(person5)
    dept1.getAll()
    dept1.writeStats()
    