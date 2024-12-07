#%% MODULE BEGINS
module_name = 'Interface'

'''
Version: 1.0

Description:
    Interfaces between user and other modules like Individuals and Departments

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
import pickle
import Individuals as ind
import Departments as dept

#other imports
import os
import platform
import subprocess

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 'Interface Main'
def intfMain():    
    try:
        while True:
            print("==========\nMAIN MENU\n==========")
            print("1. Individuals")
            print("2. Departments")
            print("3. Quit")

            choice = input("...\nEnter your choice: ")

            if choice == '1':
                print('Accessing individuals...')
                intfInd1()
            elif choice == '2':
                print('Accessing departments...')
                intfDept1()
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
            #
    except QuitProgram:
        print("Quitting program...")
#

# 'Interface Individual 1'
def intfInd1():
    while True:
        print("============\nINDIVIDUALS\n============")
        print("1. Create individual")
        print("2. Load individual")
        print("3. Back")
        print("4. Quit")

        choice = input("...\nEnter your choice: ")

        if choice == '1':
            intfInd2()
        elif choice == '2':
            print('List of people:')
            for key in people:
                print(f'\t{key}')

            nameStr = input("Enter the name of the individual to load: ")
            
            if nameStr in people:
                intfInd3(nameStr)
            else:
                print(f"Individual '{nameStr}' not found.")
            print("Going back...")
            break
        elif choice == '4':
            raise QuitProgram
        else:
            print("Invalid choice. Please try again.")
        #
    #
#

# Interface to create individual
def intfInd2(department=None, return_to_dept=False):
    while True:
        if department:
            print(f"Adding individual to the department '{department.departmentName}'")
        else:
            print("Creating individuals...")

        nameStr = input("...\nEnter individual name (or type 'done' to finish): ")

        if nameStr.lower() == 'done':
            if return_to_dept:
                print(f"Returning to department menu for '{department.departmentName}'...")
                intfDept1()
            else:
                print("Returning to the main individual menu...")
            break

        if nameStr in people:
            if department:
                department.addIndividual(people[nameStr])
            else:
                print(f"Individual '{nameStr}' already exists. No further action required.")
            continue

        individual = ind.FitnessDataProcessing(nameStr)
        people[nameStr] = individual 

        if department:
            department.addIndividual(individual)
        #
    #
#


# Interface when loading an individual
def intfInd3(nameStr=None):
   
    if not nameStr:
        nameStr = input("...\nEnter individual name: ")
    
    try:
        current = people[nameStr]
    except KeyError:
        print(f'{nameStr} can\'t be found.')
        return

    while True:
        print("================\nINDIVIDUAL INFO\n================")
        print("1. Show stats")
        print("2. Open directory")
        print("3. Find info")
        print("4. Back")
        print("5. Quit")

        choice = input("...\nEnter your choice: ")

        if choice == '1':
            current.show_stats_for_month()
        elif choice == '2':
            currentPath = f'Output\{nameStr}'
            openDirectory(currentPath)
        elif choice == '3':
            print(f"Querying data for {nameStr}...")
            intfInd4(nameStr)
        elif choice == '4':
            print('Going back...')
            break
        elif choice == '5':
            raise QuitProgram
        else:
            print("Invalid choice. Please try again.")

        #
    #
#   
    

# Interface for querying Individuals info
def intfInd4(nameStr):

    try:
        current = people[nameStr]
    except KeyError:
        print(f'{nameStr} can\'t be found.')
        return

    dataframe_choice = input("Which dataframe would you like to query? ('steps' or 'hrv'): ").strip().lower()

    if dataframe_choice == 'steps':
        dataframe = current.df_steps
        print("You have selected the 'steps' dataframe.")
        column_choice = input("Which column would you like to query? ('dayOfMonth' or 'steps'): ").strip().lower()

        if column_choice == 'dayofmonth':
            column = 'dayOfMonth'
        elif column_choice == 'steps':
            column = 'steps'
        else:
            print("Invalid column choice. Returning to main menu.")
            return 

    elif dataframe_choice == 'hrv':
        dataframe = current.df_hrv
        print("You have selected the 'hrv' dataframe.")
        column_choice = input("Which column would you like to query? ('dayOfMonth' or 'hrv'): ").strip().lower()

        if column_choice == 'dayofmonth':
            column = 'dayOfMonth'
        elif column_choice == 'hrv':
            column = 'hrv'
        else:
            print("Invalid column choice. Returning to main menu.")
            return

    else:
        print("Invalid choice for dataframe. Please choose either 'steps' or 'hrv'.")
        return

    condition = input("Enter the condition for the query (e.g., '==', '>', '<', '>=', '<=', '!='): ").strip()

    value = input("Enter the value to compare against: ").strip()

    search_value = int(value)
    
    result = current.query_data(dataframe, column, condition, search_value)

    return result
    #
#

# 'Interface Department 1'
def intfDept1():
    while True:
        print("============\DEPARTMENTS\n============")
        print("1. Create department")
        print("2. View departments")
        print("3. Load department file")
        print("4. Back")
        print("5. Quit")

        choice = input("...\nEnter your choice: ")

        if choice == '1':
            print('Creating department...')
            intfDept2()
        elif choice == '2':
            print('List of departments:')
            for key in departments:
                print(f'\t{key}')
            intfDept3()
        elif choice == '3':
            print("Loading from pickle file...")
        elif choice == '4':
            print("Going back...")
            break
        elif choice == '5':
            raise QuitProgram
        else:
            print("Invalid choice. Please try again.")
        #
    #
#

# Interface to create department
def intfDept2():
    deptNameStr = input("...\nEnter department name: ")
    days = input("Enter number of days in the month: ")

    department = dept.DepartmentDataProcessing(deptNameStr, days)
    departments[deptNameStr] = department

    print(f"Department '{deptNameStr}' created.")
    
    intfInd2(department, return_to_dept=True)
    #
#

# Method to diplay stats data
def intfDeptData(current):
   
    print("==== DATA LIST ====")
    
    try:
        print(f"Step Stats:\n{current.df_stats_steps}\n")
        print(f"HRV Stats:\n{current.df_stats_hrv}\n")
        print(f"Fitness Score:\n{current.df_stats_fitness_score}\n")
        print(f"Age Stats:\n{current.df_stats_age}\n")
        print(f"Joint Counts:\n{current.df_jointCounts}\n")
        print(f"Joint Probabilies:\n{current.df_jointProbs}\n")
        print(f"Conditional Probabilities:")
        for key, value in current.condProbs.items():
            print(f"\t{key} = {value}")

    except AttributeError as e:
        print(f"Error: {e}. Ensure the object has all the required methods.")
    #
    
#

# Interface when loading a department
def intfDept3():
    nameStr = input("...\nEnter department name: ")
    
    try:
        current = departments[nameStr]
    except KeyError:
        print(f'{nameStr} can\'t be found.')

    while True:
        print("================\nDEPARTMENT INFO\n================")
        print("1. Show stats")
        print("2. Personnel")
        print("3. Export/Import data")
        print("4. Back")
        print("5. Quit")

        choice = input("...\nEnter your choice: ")

        if choice == '1':
            print("Showing stats...")
            current.getAll()
            intfDeptData(current) 
        elif choice == '2':
            print("Accessing personnel for department...")

            if hasattr(current, 'individuals') and current.individuals:
                print(f"Personnel in department '{nameStr}':")
                for individual in current.individuals:
                    print(f"- {individual}") 
                
                nameStr = input("Enter the name of the individual that you'd like to view: ")

                if nameStr in current.individuals:
                    print(f"Accessing information for {nameStr}...")
                    intfInd3(nameStr) 
                else:
                    print(f"{nameStr} is not found in this department. Please try again.")
            else:
                print(f"No personnel data found for department '{nameStr}'.")
        elif choice == '3':
            print("Exporting/Importing data...")
            print("1. Export department data to a file")
            print("2. Import department data from a file")
            print("3. Back")
            print("4. Quit")
            sub_choice = input("...\nEnter your choice: ")
            if sub_choice == '1':
                file_name = input("Enter file name to save department data: ")
                folder_path = f'Output\{nameStr}'
                if not file_name.endswith(".pkl"):
                    file_name += ".pkl"
                file_path = os.path.join(folder_path, file_name)
                try:
                    if not current or not hasattr(current, 'departmentName'):
                        raise ValueError("The department data is invalid or empty. Cannot export.")
                    with open(file_path, 'wb') as file:
                        pickle.dump(current, file)
                    print(f"Department data exported successfully to {file_name}.")
                except Exception as e:
                    print(f"An error occurred during export: {e}")
            elif sub_choice == '2':
                file_name = input("Enter a file name to import another department data: ")
                folder_path = f'Output\{nameStr}'
                if not file_name.endswith(".pkl"):
                    file_name += ".pkl"
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'rb') as file:
                        imported_dept = pickle.load(file)
                        if hasattr(imported_dept, 'departmentName'):
                            departments[imported_dept.departmentName] = imported_dept
                            print(f"Department '{imported_dept.departmentName}' imported successfully.")
                        else:
                            print("The file does not contain valid department data.")
                except FileNotFoundError:
                    print(f"File '{file_name}' not found. Please try again.")
                except Exception as e:
                    print(f"An error occurred during import: {e}")
            elif sub_choice == '3':
                print("Going back...")
                break
            elif sub_choice == '4':
                raise QuitProgram
            else:
                print("Invalid choice. Please try again")                
        elif choice == '4':
            print('Going back...')
            break
        elif choice == '5':
            raise QuitProgram
        else:
            print("Invalid choice. Please try again.")
        #
    #
#

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

people = {} 
departments = {}

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here

class QuitProgram(Exception):
    '''Custom exception to quit the program.'''
    pass

#Function definitions Start Here
def main():
    pass
#

def openDirectory(path):
    # Check the OS and use appropriate command
    if platform.system() == 'Windows':
        os.startfile(path)  # Opens the directory on Windows
    elif platform.system() == 'Darwin':  # macOS
        subprocess.run(['open', path])
    else:  # Linux/Unix
        subprocess.run(['xdg-open', path])

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    intfMain()
    
    #TEST Code
    main()