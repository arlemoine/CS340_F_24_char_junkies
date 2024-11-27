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
import Individuals
import Departments

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def interface():
    while True:
        print("\nOptions:")
        print("1. Create record for individual")
        print("2. Create department")
        print("3. Add individuals to department")
        print("4. Process department data")
        print("5. View graphs")
        print("6. Quit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print('Creating a new record...')
            Individuals.FitnessDataProcessing('eddie')
        elif choice == '2':
            print('Creating a new department...')
        elif choice == '3':
            print('Adding individual to department...')
        elif choice == '4':
            print('Processing department data...')
        elif choice == '5':
            print('Printing graphs...')
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
        #
    #

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here



#Function definitions Start Here
def main():
    pass
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()