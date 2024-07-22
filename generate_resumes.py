import numpy as np
import yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from collections import Counter
import operator as op

import os
from together import Together

#%load_ext autoreload
#%autoreload 2

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics

# TODO: do all the modified resumes reach the threshold for classification as hire/interview? 
#Modify Java Resumes - see if accepted


with open('llm_api_keys.yaml', 'r') as file:
    config = yaml.safe_load(file)

together_api_key = config['services']['together']['api_key'] # replace with openai or anthropic also in yaml file

df = pd.read_parquet('data/resumes.parquet', engine='pyarrow')

java_dev_occupation_df = df[df["Position"]=="Java Developer"]
print(f"imported data & libraries", flush=True)

def create_modified_resumes(unmodifiedresumes, num_of_resumes, originalposition):
    # example usage of Together AI 
    modifiedresumes = []
    for resume in unmodifiedresumes[0:num_of_resumes]:
        client = Together(api_key=together_api_key) 
        position = "Project Manager"
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": "Modify the following resume to help me get a "+ position+" Job:" + resume}],
        )

        #Trim out the AI conversation (I hope this faldskraewr! stuff).
        output = response.choices[0].message.content
        output = output[output.find("\n"):output.rfind('\n')]
        output = "".join("".join(output.split('\r\n')).split('\n'))
        modifiedresumes.append(output)
        print("done 1 resume", flush=True)
    with open(originalposition+'to'+position+'.txt', 'w') as f:
        for resume in modifiedresumes:
            f.write(f"{resume}\n")
    return modifiedresumes

java_to_pm_mod_resume = create_modified_resumes(list(java_dev_occupation_df['CV']), len(java_dev_occupation_df), "JavaDeveloper")
