import pandas as pd
import argparse
import old_constants
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
from qdrant_client import QdrantClient
import constants as c

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

def return_scores(cv_s_dataframe: pd.DataFrame, job_name: str, job_description: str, verbose: bool=False):
    if len(cv_s_dataframe.columns)>1:
        raise Exception("More than one column of resumes inputted. Reformat so there is only one.")
     
    cv_column_name= cv_s_dataframe.columns[-1]
    score_column_name: str = f"{cv_column_name}{job_name} Score"

    scores_df = pd.DataFrame(index=cv_s_dataframe.index)    
    scores_df[score_column_name] = pd.NA

    # Score resumes.
    score = lambda resume : get_score(input_resume = resume, job_description = job_description, verbose = verbose)
    scores_df[score_column_name] = cv_s_dataframe[cv_column_name].apply(score)
    return scores_df[[score_column_name]]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score resumes against inputted job description",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "resumes",
        type=Path,
        nargs='+',
        help="Path to one or more resume files to improve"
    )

    parser.add_argument(
        "outputdir",
        type=Path,
        help="Path to location to save output dir."
    )

    scored_against_job_desc_details = parser.add_argument_group("Job Description Details")

    scored_against_job_desc_details.add_argument(
        '--job-description',
        type=str,
        help='Job Description for prompt.'
    )
    scored_against_job_desc_details.add_argument(
        '--job-name',
        type=str,
        help='Job Description for prompt.'
    )
    args = parser.parse_args()
    args.outputdir.mkdir(parents=True, exist_ok=True)
    return args

if __name__ == "__main__":
    args=parse_args()

    input_job_desc = args.job_description if args.job_description else c.scalable_pm_job_desc
    input_job_name = args.args.job_name if args.job_name else c.scalable_pm_job_name
    
    for resume_path in args.resumes:
        output_scores = return_scores(cv_s_dataframe=pd.read_csv(str(resume_path), index_col=0), job_name=input_job_name, job_description=input_job_desc)
        
        new_file_name = "Scores_Job_Name_"+input_job_name+"_Original_File_"+resume_path.name
        new_file_name = new_file_name.replace("/", "_").replace(" ", "_")
        if output_scores is not None:
            output_scores.to_csv(str(args.outputdir)+"/"+new_file_name)
        else:
            print("No scores outputted. Refer to previous error logs.")

#Example Input
#python3 score_cv.py test_folder/Modified_openai_gpt-4_Custom_test_cvs.csv test_folder 
