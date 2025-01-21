from . import constants
import pandas as pd
import re
import argparse
# from typing import Iterable
from collections.abc import Iterable   # import directly from collections for Python < 3.3

# Assign each entry in the filtered dataframe a label (0 for negative, 1 for positive, NA for neither)
def get_true_label(row, positive_positions: str | Iterable[str], 
                        positive_keywords: str | Iterable[str], 
                        negative_positions: str | Iterable[str], 
                        negative_keywords: str | Iterable[str],
                        verbose: bool = False):
# def get_true_label(row, positive_position: str, positive_keyword: str, negative_position: str, negative_keyword: str):

    '''
    Given a row of the dataframe, returns 
        1 if the entry belongs to the positive class
        0 if the entry belongs to the negative class
        NA if the entry is to be excluded
    Can be thought of as "h" (although this function does not operate on a feature vector)

    Currently, the positive class are entries where
    1. The primary keyword is "Project manager" (case insensitive) AND  
    2. The position contains "Project manager" (case insensitive),
    while the negative class are entries where
    1. The primary keyword is "Java Developer" (case insensitive) AND  
    2. The position contains "Java Developer" (case insensitive),

    Examine the effect of the second condition with
    print(labeled_df.loc[ (labeled_df["True Label"] == NEGATIVE_LABEL) & (labeled_df["Primary Keyword"] == PM) ])
    '''

    # Join potentially multiple phrase to form a single phrase
    if isinstance(positive_positions, str):
        positive_position = positive_positions
    else:
        positive_position = "|".join(positive_positions)

    if isinstance(positive_keywords, str):
        positive_keyword = positive_keywords
    else:
        positive_keyword = "|".join(positive_keywords)


    if isinstance(negative_positions, str):
        negative_position = negative_positions
    else:
        negative_position = "|".join(negative_positions)

    if isinstance(negative_keywords, str):
        negative_keyword = negative_keywords
    else:
        negative_keyword = "|".join(negative_keywords)

    # Escape forward slash
    # positive_position = positive_position.replace("/", "\/")
    # positive_keyword = positive_keyword.replace("/", "\/")
    # negative_position = negative_position.replace("/", "\/")
    # negative_keyword = negative_keyword.replace("/", "\/")

    if verbose:
        print(f"Positive position = {positive_position}")
        print(f"Positive keyword = {positive_keyword}")
        print(f"Negative position = {negative_position}")
        print(f"Negative keyword = {negative_keyword}")

    # Positive match
    positivePositionRegex = re.compile(f'.*({positive_position}).*', re.IGNORECASE)
    positivePrimaryKeywordRegex = re.compile(f'({positive_keyword})', re.IGNORECASE)

    position = row["Position"]
    primaryKeyword = row["Primary Keyword"]

    isPositivePositionMatch: bool = isinstance(position, str) and bool(positivePositionRegex.match(position))
    isPositivePrimaryKeywordMatch: bool = isinstance(primaryKeyword, str) and bool(positivePrimaryKeywordRegex.match(primaryKeyword))
    isPositiveMatch: bool = isPositivePositionMatch and isPositivePrimaryKeywordMatch

    # Negative match
    negativePositionRegex = re.compile(f'.*({negative_position}).*', re.IGNORECASE)
    negativePrimaryKeywordRegex = re.compile(f'({negative_keyword})', re.IGNORECASE)

    isNegativePositionMatch: bool = isinstance(position, str) and bool(negativePositionRegex.match(position))
    isNegativePrimaryKeywordMatch: bool = isinstance(primaryKeyword, str) and bool(negativePrimaryKeywordRegex.match(primaryKeyword))
    isNegativeMatch: bool = isNegativePositionMatch and isNegativePrimaryKeywordMatch

    if isPositiveMatch:
        return constants.POSITIVE_LABEL
    elif isNegativeMatch:
        return constants.NEGATIVE_LABEL
    else:
        return pd.NA

def add_true_label_column(df: pd.DataFrame, positive_position: str, positive_keyword: str, negative_position: str, negative_keyword: str, verbose: bool = False):
    '''
    Adds a new column to the dataframe with the true label in place
    '''
    TRUE_LABEL_COLUMN_NAME = "True Label"

    label = lambda resume : get_true_label(resume, positive_position, positive_keyword, negative_position, negative_keyword, verbose)
    df[TRUE_LABEL_COLUMN_NAME] = df.apply(label, axis = 1)

    return

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve resumes using various LLM providers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-num-resumes",
        type=int,
        help="Intended sample size of experiment."
    )

    parser.add_argument(
        "-true-false-split",
        type=float,
        help="Percentage of true resumes (i.e. 0.3 or 0.5), default should be 0.5."
    )

    args = parser.parse_args()
    
    return args

# Creates a true label column
if __name__ == "__main__":
    args = parse_args()
    print("Labeling 1/0 Labels for Data.")
    df = pd.read_parquet('data/resumes.parquet', engine='pyarrow')  # raw dataframe
    # Filter the dataframe minimum cv length
    MIN_CV_LENGTH = 500
    filtered_df = df.loc[df['CV'].dropna().apply(len) >= MIN_CV_LENGTH]
    labeled_df = filtered_df.copy()
    labeled_df["True Label"] = labeled_df.apply(get_true_label, axis=1)
    labeled_df = labeled_df[labeled_df["True Label"].notna()]    # Filter out rows whose label value is NA
    labeled_df.to_csv("Filtered_Truth_label.csv")   
    if args.num_resumes > len(labeled_df):
        raise

    cvs_formatted_for_experiments = pd.concat(
        labeled_df[labeled_df['True Label']==1][0:int(args.num_resumes*args.true_false_split)],
        labeled_df[labeled_df['True Label']==0][0:int(args.num_resumes - args.num_resumes*args.true_false_split)]
        )
    
    cvs_formatted_for_experiments = cvs_formatted_for_experiments['CV', 'True Label']
    cvs_formatted_for_experiments = cvs_formatted_for_experiments.reset_index(drop=True)
    cvs_formatted_for_experiments.to_csv("Final_Experiment_Resumes.csv", index=True)
