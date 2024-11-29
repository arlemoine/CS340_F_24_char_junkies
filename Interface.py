#%% MODULE BEGINS
module_name = 'Interface'

'''
Version: 1.0

Description:
    Interfaces between user and other modules like Individuals and Departments

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
            print("3. Open Output folder")
            print("4. Quit")

            choice = input("...\nEnter your choice: ")

            if choice == '1':
                print('Accessing individuals...')
                intfInd1()
            elif choice == '2':
                print('Accessing departments...')
            elif choice == '3':
                print('Opening Output folder...')
            elif choice == '4':
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
            indfInd3()
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
    
    people[nameStr] = ind.FitnessDataProcessing(nameStr)

# Interface when loading an individual
def indfInd3():
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
            currentPath = f'Output/{nameStr}'
            openDirectory(currentPath)
        elif choice == '3':
            print('Finding info via query...') # STILL NEED TO CREATE THIS ###########
        elif choice == '4':
            print('Going back...')
            break
        elif choice == '5':
            raise QuitProgram
        else:
            print("Invalid choice. Please try again.")
        #
    #

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

people = {}

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