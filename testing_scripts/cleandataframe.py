import pandas as pd
import re

# Assign each entry in the filtered dataframe a label (0 for negative, 1 for positive)
POSITIVE_LABEL, NEGATIVE_LABEL = 1, 0

def trueLabelFunction(row):
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

    # print(f"id = {row['id']}, Position = {row['Position']}, Primary Keyword = {row['Primary Keyword']}")
    
    POSITIVE_POSITION = "Project Manager"
    POSITIVE_KEYWORD = "Project Manager"

    NEGATIVE_POSITION = "QA Engineer"   # "Java Developer"
    NEGATIVE_KEYWORD = "QA"             # "Java"

    # Positive match
    positivePositionRegex = re.compile(f'.*{POSITIVE_POSITION}.*', re.IGNORECASE)
    isPositivePositionMatch: bool = isinstance(row["Position"], str) and bool(positivePositionRegex.match(row["Position"]))
    positivePrimaryKeywordRegex = re.compile(f'{POSITIVE_KEYWORD}', re.IGNORECASE)
    isPositivePrimaryKeywordMatch: bool = isinstance(row["Primary Keyword"], str) and bool(positivePrimaryKeywordRegex.match(row["Position"]))

    isPositiveMatch: bool = isPositivePositionMatch and isPositivePrimaryKeywordMatch

    # Negative match
    negativePositionRegex = re.compile(f'.*{NEGATIVE_POSITION}.*', re.IGNORECASE)
    isNegativePositionMatch: bool = isinstance(row["Position"], str) and bool(negativePositionRegex.match(row["Position"]))
    negativePrimaryKeywordRegex = re.compile(f'{NEGATIVE_KEYWORD}', re.IGNORECASE)
    isNegativePrimaryKeywordMatch: bool = isinstance(row["Primary Keyword"], str) and bool(negativePrimaryKeywordRegex.match(row["Position"]))

    isNegativeMatch: bool = isNegativePositionMatch and isNegativePrimaryKeywordMatch

    if isPositiveMatch:
        return POSITIVE_LABEL
    elif isNegativeMatch:
        return NEGATIVE_LABEL
    else:
        return pd.NA

# Creates a true label column
if __name__ == "__main__":
    print("Labeling 1/0 Labels for Data.")
    df = pd.read_parquet('data/resumes.parquet', engine='pyarrow')  # raw dataframe
    # Filter the dataframe minimum cv length
    MIN_CV_LENGTH = 500
    filtered_df = df.loc[df['CV'].dropna().apply(len) >= MIN_CV_LENGTH]
    labeled_df = filtered_df.copy()
    labeled_df["True Label"] = labeled_df.apply(trueLabelFunction, axis=1)
    labeled_df = labeled_df[labeled_df["True Label"].notna()]    # Filter out rows whose label value is NA
    labeled_df.to_csv("data/Filtered_Truth_label.csv")   