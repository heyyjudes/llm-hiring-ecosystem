from . import constants
import pandas as pd
import re
# from typing import Iterable
from collections.abc import Iterable   # import directly from collections for Python < 3.3

# Returns TRUE if a job matches a certain position and keywword
# Leave slot as empty string to not check
def is_match(row, positions: str | Iterable[str] = "", keywords: str | Iterable[str] = "") -> bool:
    
    if isinstance(positions, str):
        position_string = positions
    else:
        position_string = "|".join(positions)

    if isinstance(keywords, str):
        keyword_string = keywords
    else:
        keyword_string = "|".join(keywords)

    # Escape forward slash
    # positive_position = positive_position.replace("/", "\/")
    # positive_keyword = positive_keyword.replace("/", "\/")
    # negative_position = negative_position.replace("/", "\/")
    # negative_keyword = negative_keyword.replace("/", "\/")

    position_regex = re.compile(f'.*({position_string}).*', re.IGNORECASE)
    keyword_regex = re.compile(f'.*({keyword_string}).*', re.IGNORECASE)    # Searches for contaiment (in case of empty string)

    position = row["Position"]
    keyword = row["Primary Keyword"]
    is_position_match: bool = isinstance(position, str) and bool(position_regex.match(position))
    is_keyword_match: bool = isinstance(keyword, str) and bool(keyword_regex.match(keyword))

    return is_position_match and is_keyword_match


# Assign each entry in the filtered dataframe a label (0 for negative, 1 for positive, NA for neither)
def get_true_label(row, positive_positions: str | Iterable[str], 
                        positive_keywords: str | Iterable[str], 
                        negative_positions: str | Iterable[str], 
                        negative_keywords: str | Iterable[str],
                        verbose: bool = False):
    '''
    Given a row of the dataframe, returns 
        1 if the entry belongs to the positive class
        0 if the entry belongs to the negative class
        NA if the entry is to be excluded
    Can be thought of as "h" (although this function does not operate on a feature vector)

    Currently, the positive class are entries where
    1. The primary keyword contains "Project manager" (case insensitive) AND  
    2. The position contains "Project manager" (case insensitive),
    while the negative class are entries where
    1. The primary keyword contains "Java Developer" (case insensitive) AND  
    2. The position contains "Java Developer" (case insensitive),

    Examine the effect of the second condition with
    print(labeled_df.loc[ (labeled_df["True Label"] == NEGATIVE_LABEL) & (labeled_df["Primary Keyword"] == PM) ])
    '''

    # if verbose:
    #     print(f"Positive position = {positive_position}")
    #     print(f"Positive keyword = {positive_keyword}")
    #     print(f"Negative position = {negative_position}")
    #     print(f"Negative keyword = {negative_keyword}")

    if is_match(row, positive_positions, positive_keywords):
        return constants.POSITIVE_LABEL
    elif is_match(row, negative_positions, negative_keywords):
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


# Creates a true label column
if __name__ == "__main__":
    print("Labeling 1/0 Labels for Data.")
    df = pd.read_parquet('data/resumes.parquet', engine='pyarrow')  # raw dataframe
    # Filter the dataframe minimum cv length
    MIN_CV_LENGTH = 500
    filtered_df = df.loc[df['CV'].dropna().apply(len) >= MIN_CV_LENGTH]
    labeled_df = filtered_df.copy()
    labeled_df["True Label"] = labeled_df.apply(get_true_label, axis=1)
    labeled_df = labeled_df[labeled_df["True Label"].notna()]    # Filter out rows whose label value is NA
    labeled_df.to_csv("data/Filtered_Truth_label.csv")   