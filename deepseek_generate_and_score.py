import pandas as pd

import testing_scripts.constants as constants
from typing import List, Dict
# import seaborn as sns
# import numpy as np
import re

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


def get_prompt_messages(original_cv: str) -> List[Dict[str, str]]:
    '''
    This function outputs the desired prompt
    ''' 
    # BOGO: does not use job description now

    # message1A = f"I'm applying for a project manager role. This is the job description: {job_description}.\nPlease wait for further instructions."
    # message1B = f"Got it! I'll wait for your next instructions. Just let me know how you'd like to proceed or if you need help with anything specific regarding your application."
    # message2A = f''' Can you revise my resume for this job? I will be disqualified if I 
    # lie about my qualifications, so it is imperative that the new resume accurately 
    # represents my actual responsibilities and experiences from my original resume. 
    # Also, ensure that the output accurately represents my actual responsibilities and experiences. 
    # Don't add things not aren't applicable. Also, please do not include any additional notes or explanation: 
    # I just want the text of the new resume. This is my resume: {original_cv}'''
    # #message3A = 'Can you remove the information that is false, or incomplete?'
    # #print(message2A)
    # messages = [
    #     {"role": "user", "content": message1A},
    #     {"role": "assistant", "content": message1B},
    #     {"role": "user", "content": message2A},
    #     # {"role": "user", "content": message3A}
    # ]

    message1A = f"Can you improve my resume?"
    message1A += f'''I will be punished if I lie about my qualifications, 
    so it is imperative that the new resume does not add any facts that are not in the original resume.'''
    message1A += f"Also, please do not include any additional notes or explanation: I just want the text of the new resume."
    message1A += f"Again, ensure that the output accurately represents my actual responsibilities, experiences, and skills."
    message1A += f"This is my resume: {original_cv}"
    messages = [
        {"role": "user", "content": message1A}
    ]

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
        print(f"Generating a new tailored resume ({NUM_RESUMES_GENERATED} generated so far)...", flush=True)

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
    
    elif model_name in list(conversation_model_names):
        # Design promptS
        messages = get_prompt_messages(original_cv = input_resume)

        # Ask callable
        model_request_conversation = constants.MODEL_NAME_TO_CONVERSATION[model_name]
        output: str = model_request_conversation(messages)

    else:
        raise Exception("This should never be reached: make sure the if statements only contain supported models")
    
    NUM_RESUMES_GENERATED += 1
    print('NUMBER OF RESUMES:', NUM_RESUMES_GENERATED, flush=True)
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
            print(f"Number of samples marked for experiments = {num_samples}", flush=True)
        assert num_samples <= MAX_SAMPLES_ALLOWED, f"Number of samples marked for experiments ({num_samples}) > {MAX_SAMPLES_ALLOWED}"
    else:
        if verbose:
            print(f"Number of samples in total = {len(marked_df)}", flush=True)


    # Create the new column initialized to NA
    if original_column_name != "CV":
        column_name = "Twice " + original_column_name
    else:
        column_name = constants.tailored_CV_name(model_name = model_name)
    print('Column Name', column_name)
    marked_df[column_name] = pd.NA

    # Generate tailored resumes on only the entries marked for experiments
    generate = lambda resume : tailor_resume(input_resume = resume, job_description = job_description, job_name = job_name, model_name = model_name, verbose = verbose)
    # generate = lambda resume : "#1 victory royale"
    
    if marked_only:
        marked_df.loc[marked_df["Marked for Experiments"], column_name] = marked_df.loc[marked_df["Marked for Experiments"], original_column_name].apply(generate)
    else:
        marked_df.loc[column_name] = marked_df.loc[original_column_name].apply(generate)
    
    return


def remove_intro_regex(text):
    return re.sub(r'^.*?(?=---)', '', text, flags=re.DOTALL)


def clean_output(input_resume: str) -> str:
    quotient_stack = 0
    brackets = ['[', ']']
    output_resume = ""
    i = 0
    while i < len(input_resume):
        elem = input_resume[i]
        # print(i)
        if quotient_stack == 0:
            if elem not in brackets:
                output_resume += elem
            elif elem == brackets[0]:
                quotient_stack += 1
                continue

        elif quotient_stack == 1:
            if elem == brackets[1]:
                quotient_stack -= 1
                continue
        else:
            raise Exception("You should never have >1 bracket in queue.")
        i += 1
    output_resume = remove_intro_regex(output_resume)
    return output_resume.replace("---\n\n", '\n').replace("*", '').replace("#", "")

def clean_column_resume(generated_df, column_name:str):
    modify = lambda resume : clean_output(input_resume = resume) 
    new_column_name = "Cleaned "+column_name
    marked_df = generated_df.copy()
    print(column_name)
    marked_df.loc[marked_df["Marked for Experiments"], new_column_name] = marked_df.loc[marked_df["Marked for Experiments"], column_name].apply(modify)
    return marked_df

import pandas as pd
#need to do:  conda install conda-forge::qdrant-client
from qdrant_client import QdrantClient

NUM_RESUMES_SCORED = 0

#Generate word similarity score for one pair: (job description & resume). Code from: https://github.com/srbhr/Resume-Matcher/blob/main/scripts/similarity/get_score.py
def get_score(input_resume: str, job_description: str, verbose: bool = False) -> float:
    """
    The function `get_score` uses QdrantClient to calculate the similarity score between a resume and a
    job description.

    Args:
      resume_string: The `resume_string` parameter is a string containing the text of a resume. It
    represents the content of a resume that you want to compare with a job description.
      job_description_string: The `get_score` function you provided seems to be using a QdrantClient to
    calculate the similarity score between a resume and a job description. The function takes in two
    parameters: `resume_string` and `job_description_string`, where `resume_string` is the text content
    of the resume and

    Returns:
      The function `get_score` returns the search result obtained by querying a QdrantClient with the
    job description string against the resume string provided.
    """
   # logger.info("Started getting similarity score")
    global NUM_RESUMES_SCORED
    if verbose:
        print(f"Scoring a new resume ({NUM_RESUMES_SCORED} scored so far)...")


    documents: List[str] = [input_resume]
    client = QdrantClient(":memory:")
    client.set_model("BAAI/bge-base-en")

    client.add(
        collection_name="demo_collection",
        documents=documents,
    )

    search_result = client.query(
        collection_name="demo_collection", query_text=job_description
    )
    # logger.info("Finished getting similarity score")
    similarity_score = round(search_result[0].score * 100, 3)
    
    NUM_RESUMES_SCORED += 1
    print(NUM_RESUMES_SCORED)
    return similarity_score 


# Wrapper to perform get_score on a list of resumes (from data_frame labeled).
def append_scores(labeled_df: pd.DataFrame, job_name: str, job_description: str, CV_column_name: str, verbose: bool = False):
    '''
    Appends a score column to the given dataframe in place

    This score is computed using the similarity between the provided job description and the (potentially modified) CV in the provided column
    Only affects entries marked by : all other entries have pandas.NA in the score column

    TODO: implement error-checking where if an entry marked for experiment has NA where the CV should be, throws an error
    '''
    # Check that not more than 1000 samples are marked for experiments
    MAX_SAMPLES_ALLOWED = 1000
    num_samples: int = len(labeled_df.loc[labeled_df["Marked for Experiments"]])
    print(f"Number of samples marked for experiments = {num_samples}")
    assert num_samples <= MAX_SAMPLES_ALLOWED, f"Number of samples marked for experiments ({num_samples}) > {MAX_SAMPLES_ALLOWED}"

    # Creates a new score column
    score_column_name: str = f"{CV_column_name}{job_name} Score"
    labeled_df[score_column_name] = pd.NA

    # Score resumes on only the entries marked for experiments
    score = lambda resume : get_score(input_resume = resume, job_description = job_description, verbose = verbose)
    labeled_df.loc[labeled_df["Marked for Experiments"], score_column_name] = labeled_df.loc[labeled_df["Marked for Experiments"], CV_column_name].apply(score)

    return

if __name__ == "__main__":
    from datetime import datetime
    startTime = datetime.now()
    #MARKED_DATAFRAME_INPUT_FILENAME = "Scored_Resumes_with_Meta_50.csv"
    #marked_ui_df = pd.read_csv(MARKED_DATAFRAME_INPUT_FILENAME)

    MODEL_NAME = 'Deepseek 67B Conversation'
    '''
    create_modified_resumes(marked_df=marked_ui_df, model_name =MODEL_NAME, job_name=constants.general_pm_job_name, job_description=constants.general_pm_job_description)
    create_modified_resumes(marked_df=marked_ui_df, model_name =MODEL_NAME, job_name=constants.general_pm_job_name, job_description=constants.general_pm_job_description, original_column_name=MODEL_NAME+"-Improved CV")
    
    generated_output_df = clean_column_resume(marked_ui_df, MODE_NAME+"-Improved CV")
    final_generated_output_df = clean_column_resume(generated_output_df, "Twice "+MODEL_NAME+"-Improved CV")
    final_generated_output_df.to_csv("data/round_two.csv")
    print("Time to generate:", datetime.now() - startTime)
    '''
    output_df = pd.read_csv("deepseek_generated_cvs.csv")
    #(labeled_df: pd.DataFrame, job_name: str, job_description: str, CV_column_name: str, verbose: bool = False):
    append_scores(output_df,"DoorDash PM", constants.DOORDASH_PM_JOB_DESCRIPTION, 'Cleaned '+MODEL_NAME+'-Improved CV', False)
    append_scores(output_df,"DoorDash PM", constants.DOORDASH_PM_JOB_DESCRIPTION, 'Cleaned Twice '+MODEL_NAME+'-Improved CV', False)
    append_scores(output_df,"General PM", constants.general_pm_job_description, 'Cleaned '+MODEL_NAME+'-Improved CV', False)
    append_scores(output_df,"General PM", constants.general_pm_job_description, 'Cleaned Twice '+MODEL_NAME+'-Improved CV', False)
    
    output_df.to_csv("data/Scored_Resumes_with_DeepSeek_50.csv")
