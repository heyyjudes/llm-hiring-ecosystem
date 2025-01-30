# import numpy as np
# import yaml
# import os

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
    print(NUM_RESUMES_SCORED, flush=True)
    return similarity_score 


# Wrapper to perform get_score on a list of resumes (from data_frame labeled).
def append_scores(labeled_df: pd.DataFrame, job_description: str, job_description_name: str, CV_column_name: str, verbose: bool = False):
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
    score_column_name: str = f"{CV_column_name} {job_description_name} Score"
    labeled_df[score_column_name] = pd.NA

    # Score resumes on only the entries marked for experiments
    score = lambda resume : get_score(input_resume = resume, job_description = job_description, verbose = verbose)
    print(CV_column_name)
    labeled_df.loc[labeled_df["Marked for Experiments"], score_column_name] = labeled_df.loc[labeled_df["Marked for Experiments"], CV_column_name].apply(score)

    return
    
    # #appends score of similarity of Job title to dataframe.
    # scores = [0 for _ in range(len(labeled_df))]
    # counter = 0
    # print("enter function", flush=True)
    
    # cv_label = 'CV'
    # column_title = job_title+" Score"
    # if is_modified:
    #     cv_label = job_title + ' Modified CV'
    #     column_title = job_title + " Modified Resume Score"

    # for index, row in dataframe_labeled.iterrows():
    #     resumm = "".join(row.to_dict()['CV'].split('\r\n')).split('\n')
    #     similarity_score = get_score("".join(resumm), job_description)
    #     scores[counter] = similarity_score
    #     print(similarity_score, row['Position'], flush=True)
    #     counter+=1
    
    # dataframe_labeled[column_title] = scores
    # #Returns dataframe with extra column indicating position-similarity score.
    # return dataframe_labeled


if __name__ == "__main__":
    print("Scoring Data for resumes.", flush=True)
    #from cleandataframe import trueLabelFunction
    # Filter the dataframe minimum cv length
    MIN_CV_LENGTH = 500
    labeled_df = pd.read_csv("validation_tests/final_paper_classification_outputs_doordash.csv")
    labeled_df = labeled_df[labeled_df["True Label"].notna()] 
    with open('sample_input_data/example_job_descriptions/scalable_job_description.txt', 'r') as file:
      scalable_content = file.read()

    columns = ['CV', 'Cleaned GPT-4o Conversation-Improved CV', 'Cleaned Twice GPT-4o Conversation-Improved CV']

    for col in columns:
      append_scores(labeled_df, scalable_content, " Scalable PM", col)
    
    labeled_df.to_csv("data/scalable_two.csv")