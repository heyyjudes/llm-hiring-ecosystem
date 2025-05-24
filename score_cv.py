'''
This module provides functions to score resumes/CVs against inputted job descriptions.

Running score_cv takes the following three inputs - and outputs the scores of the inputted CVs in .csv form:
1. Input CVs (Filepath(s), Required)
2. Output Directory (Filepath, Required)
3. Job Description (String, Optional) - defaults to "DoorDash PM" Job Description.
4. Job Name (String, Optional) - defaults to "DoorDash PM" Job Description

# Example Input
# make a folder score_resumes where you put the resumes you want to score
# python3 score_cv.py --resume-folder score_resumes --job-name DD_PM --job-description doordash_pm.txt --output-dir test_folder

'''
import os

import pandas as pd
import argparse
from typing import List
from pathlib import Path
from qdrant_client import QdrantClient
import constants as c
from tqdm import tqdm
tqdm.pandas()

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
    client = QdrantClient(path="./local_model")
    client.set_model("BAAI/bge-base-en")

    if client.get_collections().collections:
        client.delete_collection(collection_name="demo_collection")

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
    return similarity_score 

def return_scores(cv_s_dataframe: pd.DataFrame, job_name: str, job_description: str, verbose: bool=False) -> pd.DataFrame:
    if len(cv_s_dataframe.columns)>1:
        raise Exception("More than one column of resumes inputted. Reformat so there is only one.")
     
    cv_column_name= cv_s_dataframe.columns[-1]
    score_column_name: str = f"{cv_column_name} {job_name} Score"

    scores_df = pd.DataFrame(index=cv_s_dataframe.index)    
    scores_df[score_column_name] = pd.NA

    # Score resumes.
    score = lambda resume : get_score(input_resume = resume, job_description = job_description, verbose = verbose)
    scores_df[score_column_name] = cv_s_dataframe[cv_column_name].progress_apply(score)
    return scores_df[[score_column_name]]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score resumes against inputted job description",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--resumes",
        type=Path,
        nargs='+',
        help="Path to one or more resume files to score"
    )

    parser.add_argument(
        "--resume-folder",
        type=str,
        default=None,
        help="Path to one or more resume files to score"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to location to save output dir."
    )

    scored_against_job_desc_details = parser.add_argument_group("Job Description Details")

    scored_against_job_desc_details.add_argument(
        '--job-description',
        type=Path,
        help='Job Description to be scored against.'
    )
    scored_against_job_desc_details.add_argument(
        '--job-name',
        type=str,
        help='Job Name to be scored against.'
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args

if __name__ == "__main__":
    args=parse_args()
    
    input_job_name = args.job_name if args.job_name else c.doordash_pm_job_name
    input_job_desc = open(args.job_description, 'r').read() if args.job_description else c.doordash_pm_job_desc

    resumes = args.resumes
    if args.resume_folder is not None:
        folder_path = Path(args.resume_folder)
        resumes = os.listdir(args.resume_folder)
        resumes = [folder_path / r for r in resumes if r.endswith(".csv")]
        print(f"scoring entire folder: {folder_path}")

    output_df_list = []
    for resume_path in resumes:
        print(resume_path)
        input_csv = pd.read_csv(str(resume_path), index_col='id')
        output_df = return_scores(cv_s_dataframe=input_csv,
                                      job_name=input_job_name,
                                      job_description=input_job_desc)
        output_df_list.append(output_df)

    if len(output_df_list) < 1:
        print("No scores outputted. Refer to previous error logs.")
    else:
        if len(output_df_list) == 1:
            new_file_name = "Scores"+input_job_name+"_Original_File_"+resume_path.name
            new_file_name = new_file_name.replace("/", "_").replace(" ", "_")
            final_df = output_df_list[0]
        else:
            final_df = pd.concat(output_df_list, axis=1)
            new_file_name = "-".join(["Scores", input_job_name, f"{folder_path.name}.csv"])
            new_file_name = new_file_name.replace("/", "_").replace(" ", "_")

        final_df.to_csv(str(args.output_dir)+"/"+new_file_name)