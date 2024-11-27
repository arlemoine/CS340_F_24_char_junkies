#Version: v0.1
#Date Last Updated: 12-20-2023

#%% MODULE BEGINS
module_name = 'Logging'

'''
Version: 1.0

Description:
    This module is used to provide logging functionality to the other modules.

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

#other imports
import logging
import os

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Function definitions Start Here

def configure_logger(name, log_dir='Output/Log', level=logging.INFO):
    '''
    Configures and returns a logger for the given name.

    Returns: Logging.logger: Configured logger instance.
    '''
    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create and configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create file handler
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create and set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers if logger is reused
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
#

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()
#