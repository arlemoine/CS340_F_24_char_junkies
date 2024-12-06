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
            print('Creating individual...')
            intfInd2()
        elif choice == '2':
            print('List of people:')
            for key in people:
                print(f'\t{key}')
            intfInd3()
        elif choice == '3':
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
def intfInd2(): 
    nameStr = input("...\nEnter individual name: ")
    filepath = f'Input/{nameStr}'
    
    # Ensure the person has an input folder before creating
    if not os.path.isdir(filepath):
        print("Person doesn't exist")
        return
    #
    
    people[nameStr] = ind.FitnessDataProcessing(nameStr)
#


# Interface when loading an individual
def intfInd3():
    nameStr = input("...\nEnter individual name: ")
    
    # Ensure the person exists before accessing the menu below
    try:
        current = people[nameStr]
    except KeyError:
        print(f'{nameStr} can\'t be found.')

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
            indfInd4(nameStr)
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
def indfInd4(nameStr):

    # Ensure the person exists before accessing the data
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
    days = input("Enter number of days in month: ")
    departments[deptNameStr] = dept.DepartmentDataProcessing(deptNameStr, days)
    print("Department created.")
#

# Interface when loading a department
def intfDept3():
    nameStr = input("...\nEnter department name: ")
    
    # Ensure the department exists before accessing the menu below
    try:
        current = departments[nameStr]
    except KeyError:
        print(f'{nameStr} can\'t be found.')

    while True:
        print("================\nDEPARTMENT INFO\n================")
        print("1. Show stats")
        print("2. Open directory")
        print("3. Personnel")
        print("4. Export/Import data")
        print("5. Back")
        print("6. Quit")

        choice = input("...\nEnter your choice: ")

        if choice == '1':
            print("Showing stats...")
        elif choice == '2':
            print("Opening directory...")
            # currentPath = f'Output/{nameStr}'
            # openDirectory(currentPath)
        elif choice == '3':
            print("Accessing personnel for department...")
        elif choice == '4':
            print("Exporting/Importing data...")
        elif choice == '5':
            print('Going back...')
            break
        elif choice == '6':
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