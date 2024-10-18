# Imports 
import os
import pandas as pd
import testing_scripts.constants as constants
import numpy as np
import matplotlib.pyplot as plt
from testing_scripts import generate_resumes
from testing_scripts import score_resumes

#Generate input data with running script - label_resumes.py found in folder "testing_scripts."
input_data = pd.read_csv("data/_3000_testing_data.csv")

#Name of Folder to store data.
saved_folder = "data/10_test_data/"

#Set Constants
start = 0 #Set index of starting point of script. (Starts at 0th resume.) 
resumes_index=100 #Generates and scores scripts in batches of 100.
end = start+resumes_index #End of Batch
input_model = "Together Conversation"
end_script = 3000 #Final Number of Resumes - stops once reaches this point.

#Name and description of job to be scored against.
input_job_name ='Doordash PM'
input_job_description = constants.DOORDASH_PM_JOB_DESCRIPTION

#Generates LLM-modified resumes and scores resumes in batches. Data saved in batches.
#We process in batches as this script takes a long time to run! So this helps us start & restart the script as neccesary.
while end < end_script:
    print("Starting:", start, end, flush=True)

    #Select section of data to be parsed.
    saved_data = input_data[start:end] 

    #Generate & clean once modified resumes.
    generate_resumes.create_modified_resumes(saved_data, job_name = constants.JOB_NAME, job_description = constants.JOB_DESCRIPTION, model_name = input_model, verbose = True, original_column_name='CV')
    saved_data = generate_resumes.clean_column_resume(saved_data, input_model+'-Improved CV')
   
    #Generate & clean Twice modified resumes.
    generate_resumes.create_modified_resumes(saved_data, job_name = constants.JOB_NAME, job_description = constants.JOB_DESCRIPTION, model_name = input_model, verbose = True, original_column_name=input_model+'-Improved CV')
    saved_data = generate_resumes.clean_column_resume(saved_data, 'Twice '+input_model+'-Improved CV')

    #Score original resumes, once modified resumes, and twice modified resumes.
    score_resumes.append_scores(saved_data, input_job_name, input_job_description, 'CV', False)
    score_resumes.append_scores(saved_data, input_job_name, input_job_description, 'Cleaned Together Conversation-Improved CV', False)
    score_resumes.append_scores(saved_data, input_job_name, input_job_description, 'Cleaned Twice Together Conversation-Improved CV', False)
    saved_data.to_csv(saved_folder+"_generated_resumes_data+{start}_to_{end}.csv")
   
    #Index to next batch
    start=end
    end+=resumes_index
    print("DONE:", start, flush=True)

#To indicate batch processing is done.

#Combines all outputted data into one csv.
def combine_dataframes(folder_name, output_file_name):
    allfiles = list(os.listdir(folder_name))
    return_file = pd.read_csv(allfiles[0])
    for elem in allfiles[1:]:
        return_file  = return_file._append(pd.read_csv(elem))
    return_file.to_csv(output_file_name)
    return return_file

combine_dataframes(saved_folder, 'output_combined_test_data.csv')

print("Saved All Csvs.", flush=True)
