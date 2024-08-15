import pandas as pd
from . import constants
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

def tailor_resume(input_resume: str, job_description: str, model_name: str, verbose: bool = False) -> str:
    '''
    This function accepts an input resume, a job description, and a model name (since different models may require different prompts)
    The function uses the model to tailor the resume toward the job description
        The function fails an assertion if the inputed model is not one of the listed models
    '''
    global NUM_RESUMES_GENERATED
    
    assert model_name in constants.MODEL_NAME_TO_CALLABLE.keys(), f"Error: model_name ({model_name}) must be in {constants.MODEL_NAME_TO_CALLABLE.keys()}"

    if verbose:
        print(f"Generating a new tailored resume ({NUM_RESUMES_GENERATED} generated so far)...")

    if model_name == "Together" or model_name == "OpenAI" or model_name == "GPT-4o-mini":
        # Design prompt
        # prompt: str = f"Tailor my resume to this job description and do not make anything up. It is imperative that you provide only the content of the CV without any additional explanations, notes, or comments."
        # prompt += f" This is the job description: {job_description}"
        # prompt += f" This is my resume: {input_resume}"

        prompt: str = f"Improve the following resume for a project manager job. It is imperative that you do not make any information or qualifications up and that you provide only the content of the CV without any additional explanations, notes, or comments.\n"
        prompt += f"ORIGINAL RESUME: {input_resume}"

    else:
        raise Exception("This should never be reached: make sure the if statements only contain supported models")
    
    # Ask the model
    model_request_callable = constants.MODEL_NAME_TO_CALLABLE[model_name]
    output: str = model_request_callable(prompt)
    NUM_RESUMES_GENERATED += 1
    return output

def create_modified_resumes(marked_df, model_name: str, job_name: str, job_description: str, verbose: bool = False, marked_only: bool = True) -> str:
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
    column_name: str = f"{model_name}-Improved {job_name} CV"
    marked_df[column_name] = pd.NA

    # Generate tailored resumes on only the entries marked for experiments
    generate = lambda resume : tailor_resume(input_resume = resume, job_description = job_description, model_name = model_name, verbose = verbose)
    # generate = lambda resume : "#1 victory royale"
    
    if marked_only:
        marked_df.loc[marked_df["Marked for Experiments"], column_name] = marked_df.loc[marked_df["Marked for Experiments"], "CV"].apply(generate)
    else:
        marked_df.loc[column_name] = marked_df.loc["CV"].apply(generate)
    
    return

if __name__ == "__main__":
    # Import the dataframe without the generated resumes
    print("Starting reading labeled dataframe...")
    labeled_df = pd.read_csv("data/labeled_df_without_generated_resumes.csv")
    print("Finished reading labeled dataframe...\n")
    

    # Generate the tailored resumes
    BITS_ORCHESTRA_PM_JOB_NAME = "Bits Orchestra PM"
    BITS_ORCHESTRA_PM_JOB_DESCRIPTION = "A commitment to collaborative problem solving, agile thinking, and adaptability is essential. We are looking for a candidate who is able to balance a fast moving and changing environment with the ability to identify, investigate, and predict project risks Recruiting stages: HR interview, Tech interview **Core Responsibilities:** - Manage the full project life cycle including requirements gathering, creation of project plans and schedules, obtaining and managing resources, and facilitating project execution, deployment, and closure. - In cooperation with Technical Leads create and maintain comprehensive project documentation. - Manage Client expectations, monitor and increase CSAT level; - Plan, perform and implement process improvement initiatives. - Organize, lead, and facilitate cross-functional project teams. - Prepare weekly and monthly project status reports **What you need to Succeed:** - 1+ Year of dedicated Project Management in a production environment - Excellent organization and communication skills and the ability to communicate effectively with customers and co-workers. - Strong understanding of a Project Management Methodology (SDLC, Agile, Waterfall, etc.) - Creative mind with the ability to think outside-of-the-box. - The ability to manage multiple projects simultaneously - Experience with Jira or similar project management tool - Upper-intermediate level of English is a must"
    # PAYMENT_SERVICE_PM_JOB_DESCRIPTION = "We are seeking a highly organized and experienced Senior Project Manager/Program Manager Responsibilities: - Develop and maintain project plans, including scope, timeline, resource allocation, and deliverables. - Monitor and control project progress, identify risks, and proactively implement mitigation strategies. - Lead a team of project managers, developers, testers, and other project resources. - Manage resource allocation, workload distribution, and capacity planning. - Writing technical specifications and documentation. - Communicate with partners and payment providers. - Have experience working on SAFe. Requirements: - Bachelor’s/Master’s degree in a relevant field (Computer Science, Information Technology, or similar). - Experience with acquiring, payments and gateway integration. - English — Upper intermediate. - Experience in integration of payment service providers like as Stripe, PayPal, LiqPay - Senior project manager experience with a strong history of successfully delivering engineering E-commerce projects for external clients. - Strong understanding of project management methodologies, tools, and techniques. - Excellent leadership and team management skills."
    print(f"Starting generating tailored resumes for {BITS_ORCHESTRA_PM_JOB_NAME}")
    
    create_modified_resumes(marked_df = labeled_df, 
                            model_name = "Together", 
                            job_name = BITS_ORCHESTRA_PM_JOB_NAME,
                            job_description = BITS_ORCHESTRA_PM_JOB_DESCRIPTION,
                            verbose = True)
    print(f"Finished generating tailored resumes for {BITS_ORCHESTRA_PM_JOB_NAME}\n")

    # Export
    print(f"Starting export of new dataframe")
    labeled_df.to_csv("data/labeled_df_with_generated_resumes.csv")
    print(f"Finished export of new dataframe\n")

    #  from cleandataframe import trueLabelFunction
    #  df = pd.read_parquet('data/resumes.parquet', engine='pyarrow')  # raw dataframe
    #  # Filter the dataframe minimum cv length
    #  MIN_CV_LENGTH = 500
    #  filtered_df = df.loc[df['CV'].dropna().apply(len) >= MIN_CV_LENGTH]
    #  labeled_df = filtered_df.copy()
    #  labeled_df["True Label"] = labeled_df.apply(trueLabelFunction, axis=1)
    #  labeled_df = labeled_df[labeled_df["True Label"].notna()] 

    #  generated_resumes = create_modified_resumes(labeled_df, len(labeled_df), 'Project Manager')
    #  generated_resumes.to_csv("data/withgeneratedresumes.csv")
    