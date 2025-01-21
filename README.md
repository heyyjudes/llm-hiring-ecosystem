## Algorithmic Hiring in LLM Ecosystems 

This repository contains code to first: modify resumes with existing large language models (found in modify_cv.py) and score them against existing job descriptions (score_cv.py).

### Setup

To install environment:

```
conda env create -f env.yml
```

### Modifying and Scoring Resumes

modify_cv.py is a Python script that improves/modifies resumes/CVs using various LLM APIs given a set of inputted resumes and custom prompts.

#### Inputs and Outputs for modify_cv.py

Running modify_cv takes in the following inputs and outputs the modified CVs in a .csv file. Optional inputs also have default values in the code:
 
##### Input Parameters

1. **Input CVs**  
   - **Type**: Filepath(s)  
   - **Required**

2. **Output Directory**  
   - **Type**: Filepath  
   - **Required**

3. **Prompt Template**  
   - **Type**: Filepath (`.json` or `.txt`)  
   - **Required**  
   - **Details**: Prompt with a placeholder for `{original_cv}` (optional placeholder for `{job_description}`). We recommend using '.txt' files if you only need to interface with the LLM-API as a "user". Otherwise, check the example-json prompt template to interface with the LLM-API as an assistant (in addition to user).

4. **Job Description for Prompt**  
   - **Type**: `.txt`  
   - **Optional**  
   - **Details**: User-inputted prompt for the last prompt type.

5. **LLM Provider**  
   - **Type**: String  
   - **Options**: `OpenAI`, `Together`, `Anthropic`  
   - **Required**

6. **API-Key**  
   - **Type**: Filepath  
   - **Required**  
   - **Details**: Path to the `api_keys.yaml` file.

7. **Model**  
   - **Type**: String  
   - **Optional**  
   - **Details**: Name of the model to use (besides default).

It outputs a csv, timestamped, with one column corresponding to the modified resume/CV text. 

To test modify_cv.py with our example files and anti-hallucination-prompt, per described in our manuscript, run in the root directory of this folder: 

```
python3 modify_cv.py sample_input_data/example_input_cvs/three_example_cvs.csv sample_input_data/example_output_data --prompt-template sample_input_data/example_prompts/anti_hallucination_llm_prompt.txt --prompt-job-description sample_input_data/example_job_descriptions/scalable_job_description.txt --provider openai --api-key llm_api_keys.yaml 
```

#### Inputs and Outputs for `score_cv`

The `score_cv` function takes the following inputs and outputs the scores of the inputted CVs in `.csv` format:

1. **Input CVs**  
   - **Type**: Filepath(s)  
   - **Required**

2. **Output Directory**  
   - **Type**: Filepath  
   - **Required**

3. **Job Description**  
   - **Type**: String  
   - **Optional**  
   - **Default**: `"Scalable"` Job Description (see `sample_input_data/example_job_descriptions`).

4. **Job Name**  
   - **Type**: String  
   - **Optional**  
   - **Default**: `"Scalable"` Job Description (see `sample_input_data/example_job_descriptions`).

It is natural to run score_cv.py on the output resumes of modify_cv.py (and input resumes too). To test score_cv.py with our example files, run in the root directory of this folder: 

```
python3 score_cv.py sample_input_data/example_input_cvs/three_example_cvs.csv sample_input_data/example_output_data --job-description sample_input_data/example_job_descriptions/scalable_job_description.txt --job-name Scalable
```

### Analyzing Outputted Resume Scores

Code to first, compare the spreads of our resume scores across different large language models, and second, analyze how different binary threshold classifiers, subject to different FPR constraints, perform on our scored resume data can be found in validation_tests/significance_tests.ipynb. 
