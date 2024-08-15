import pandas as pd
import testing_scripts.constants as constants
from typing import List, Dict
import re
# import seaborn as sns
# import numpy as np
# import re

# %load_ext autoreload
# %autoreload 2

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import sklearn.metrics as metrics

# TODO: do all the modified resumes reach the threshold for classification as hire/interview? 
# pModify Java Resumes - see if accepted

NUM_RESUMES_GENERATED = 0

def get_prompt_messages(original_cv: str, job_description: str, job_name: str) -> List[Dict[str, str]]:
    '''
    This function outputs the desired prompt
    ''' 
    message1A = f"I'm applying for a project manager role. This is the job description: {job_description}.\nPlease wait for further instructions."
    message1B = f"Got it! I'll wait for your next instructions. Just let me know how you'd like to proceed or if you need help with anything specific regarding your application."
    message2A = f''' Can you revise my resume for this job? I will be disqualified if I 
    lie about my qualifications, so it is imperative that the new resume accurately 
    represents my actual responsibilities and experiences from my original resume. 
    Also, ensure that the output accurately represents my actual responsibilities and experiences. 
    Don't add things not aren't applicable. Also, please do not include any additional notes or explanation: 
    I just want the text of the new resume. This is my resume: {original_cv}'''
    #message3A = 'Can you remove the information that is false, or incomplete?'
    #print(message2A)
    messages = [
        {"role": "user", "content": message1A},
        {"role": "assistant", "content": message1B},
        {"role": "user", "content": message2A},
        # {"role": "user", "content": message3A}
    ]
    # print("generating..", flush=True)     # moved to tailor resume
    return messages

def tailor_resume(input_resume: str, job_description: str, job_name: str, model_name: str, verbose: bool = False) -> str:
    '''
    This function accepts an input resume, a job description, and a model name (since different models may require different prompts)
    The function uses the model to tailor the resume toward the job description
        The function fails an assertion if the inputed model is not one of the listed models

    Can now support either a single prompt (callable) or multiple prompts (conversation)
    '''
    global NUM_RESUMES_GENERATED

    # Check for inclusion (callable or conversation)
    callable_model_names = constants.MODEL_NAME_TO_CALLABLE.keys()
    conversation_model_names = constants.MODEL_NAME_TO_CONVERSATION.keys()
    assert model_name in callable_model_names or model_name in conversation_model_names, f"Error: model_name ({model_name}) must be in {callable_model_names} or {conversation_model_names}"

    if verbose:
        print(f"Generating a new tailored resume ({NUM_RESUMES_GENERATED} generated so far)...")

    if model_name == "Together" or model_name == "GPT-4o" or model_name == "GPT-4o-mini":
        # Design prompt
        # prompt: str = f"Tailor my resume to this job description and do not make anything up. It is imperative that you provide only the content of the CV without any additional explanations, notes, or comments."
        # prompt += f" This is the job description: {job_description}"
        # prompt += f" This is my resume: {input_resume}"

        prompt: str = f"Improve the following resume for a project manager job. It is imperative that you do not make any information or qualifications up and that you provide only the content of the CV without any additional explanations, notes, or comments.\n"
        prompt += f"ORIGINAL RESUME: {input_resume}"

        prompt = "Write a haiku about bugs"

        # Ask callable
        model_request_callable = constants.MODEL_NAME_TO_CALLABLE[model_name]
        output: str = model_request_callable(prompt)
    
    elif model_name == "GPT-4o Conversation" or model_name == "GPT-4o-mini Conversation":
        # Design promptS
        messages = get_prompt_messages(original_cv = input_resume, job_description = job_description, job_name = job_name)

        # Ask callable
        model_request_conversation = constants.MODEL_NAME_TO_CONVERSATION[model_name]
        output: str = model_request_conversation(messages)

    else:
        raise Exception("This should never be reached: make sure the if statements only contain supported models")
    
    NUM_RESUMES_GENERATED += 1
    return output


def create_modified_resumes(marked_df, model_name: str, job_name: str, job_description: str, verbose: bool = False, marked_only: bool = True, original_column_name: str = 'CV') -> str:
    '''
    Given a labeled_df, create a new column and generate resumes tailored the provided job using the model with the provided model name
    Only affects the entries marked for experiments
    Modifies the dataframe in place

    TODO: right now, this recreates the column every time
    Implement in a way that doesn't recreates the column
    '''
    # Check that not more than 1000 samples are marked for experiments
    if marked_only:
        MAX_SAMPLES_ALLOWED = 1000
        num_samples: int = len(marked_df.loc[marked_df["Marked for Experiments"]])
        
        if verbose:
            print(f"Number of samples marked for experiments = {num_samples}")
        assert num_samples <= MAX_SAMPLES_ALLOWED, f"Number of samples marked for experiments ({num_samples}) > {MAX_SAMPLES_ALLOWED}"
    else:
        if verbose:
            print(f"Number of samples in total = {len(marked_df)}")


    # Create the new column initialized to NA
    if original_column_name != "CV":
        column_name = "Twice" + original_column_name
    else:
        column_name = constants.tailored_CV_name(model_name = model_name, job_name = job_name)
    marked_df[column_name] = pd.NA

    # Generate tailored resumes on only the entries marked for experiments
    generate = lambda resume : tailor_resume(input_resume = resume, job_description = job_description, job_name = job_name, model_name = model_name, verbose = verbose)
    # generate = lambda resume : "#1 victory royale"
    
    if marked_only:
        marked_df.loc[marked_df["Marked for Experiments"], column_name] = marked_df.loc[marked_df["Marked for Experiments"], original_column_name].apply(generate)
    else:
        marked_df.loc[column_name] = marked_df.loc[original_column_name].apply(generate)
    
    return

def clean_output(input_resume: str)->str:
    # quotient_stack = 0
    # brackets = ['[', ']']
    # output_resume = ""
    # i=0
    # while i < len(input_resume):
    #     elem = input_resume[i]
    #     #print(i)
    #     if quotient_stack == 0:
    #         if elem not in brackets:
    #             output_resume+=elem
    #         elif elem == brackets[0]:
    #             quotient_stack +=1
    #             continue

    #     elif quotient_stack ==1:
    #         if elem == brackets[1]:
    #             quotient_stack-=1
    #             continue
    #     else:
    #         raise Exception("You should never have >1 bracket in queue.")
    #     i+=1
    
    # return output_resume.replace("\n", '').replace("*", '').replace("#", "")

    regex: str = r'\[.*\]|\*|#|\n'
    return re.sub(regex, "", input_resume)

def clean_column_resume(generated_df, column_name:str):
    modify = lambda resume : clean_output(input_resume = resume) 
    new_column_name = "Cleaned "+column_name
    marked_df = generated_df.copy()
    #print(column_name)
    marked_df.loc[marked_df["Marked for Experiments"], new_column_name] = marked_df.loc[marked_df["Marked for Experiments"], column_name].apply(modify)
    return marked_df

if __name__ == "__main__":
    from datetime import datetime
    startTime = datetime.now()
    MARKED_DATAFRAME_INPUT_FILENAME = "data/marked_df_100 PM vs 100 UI_1000 chars min.csv"
    marked_ui_df = pd.read_csv(MARKED_DATAFRAME_INPUT_FILENAME)
    create_modified_resumes(marked_df=marked_ui_df, job_name=constants.BITS_ORCHESTRA_PM_JOB_NAME, job_description=constants.BITS_ORCHESTRA_PM_JOB_DESCRIPTION)
    output_df = clean_column_resume(marked_ui_df, marked_ui_df.columns[-1])
    marked_ui_df.to_csv("data/chatgpt_generated_df_100 PM vs 100 UI_1000 chars min.csv")
    print("Time to run:", datetime.now() - startTime)
