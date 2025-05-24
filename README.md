## Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations

This repository contains code and experiments for the ICML paper [Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations](https://www.arxiv.org/abs/2502.13221).

## Abstract
In an era of increasingly capable foundation models, job seekers are turning to generative AI tools to
enhance their application materials. However, unequal access to and knowledge about generative AI tools
can harm both employers and candidates by reducing the accuracy of hiring decisions and giving some
candidates an unfair advantage. To address these challenges, we introduce a new variant of the strategic
classification framework tailored to manipulations performed using large language models, accommodating
varying levels of manipulations and stochastic outcomes. We propose a “two-ticket” scheme, where
the hiring algorithm applies an additional manipulation to each submitted resume and considers this
manipulated version together with the original submitted resume. We establish theoretical guarantees
for this scheme, showing improvements for both the fairness and accuracy of hiring decisions when the
true positive rate is maximized subject to a no false positives constraint. We further generalize this
approach to an n-ticket scheme and prove that hiring outcomes converge to a fixed, group-independent
decision, eliminating disparities arising from differential LLM access. Finally, we empirically validate our
framework and the performance of our two-ticket scheme on real resumes using an open-source resume
screening tool.

## Overview

This repository provides implementations for experiments in the paper to verify the theoretical improvements of our "two-ticket" scheme. We use part of the [Djinni Recruitment Dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english/blob/main/README.md) dataset for sample resumes, and their matched occupations (i.e. Product Manager, UI/UX designer, etc.). We also then draw from various online job postings for the relevant job descriptions we score our resumes against. 

## Data & LLM Tools

### Djinni Recruitment Dataset

The Djinni dataset file which we used to generate results for the effectiveness of our two ticket system in Table 1 ('Table1_Experimental_Modified_Resumes/Original_CV.csv) can be downloaded from:
- [Stereotypes in Recruitment Dataset](https://github.com/Stereotypes-in-LLMs/recruitment-dataset) - Downloaded All Data, and Filtered for first 260 Product Manager and first 260 UI/UX designer resumes.

The Djinni dataset file which we used to generate results for the effectiveness of different LLM tools and their performance against different job descriptions in Figure 1 (data under 'Figure1_100Samples/Resumes/original.csv') is a subset of our above data: namely we filtered for the first 50 out of 260 Product Manager and first 50 260 UI/UX designer resumes.

### Job Descriptions Data
The two job descriptions used to generate results for resume scores in Table 1 can be found here:
- [DoorDash PM Job Description]('sample_input_data/example_job_descriptions/doordash_pm.txt')  - the description was drawn directly from [here](https://careersatdoordash.com/jobs/product-manager-multiple-levels/5523275/).
- [Google UX Designer Job Description]('sample_input_data/example_job_descriptions/google_ux.txt') - the job description has since been taken down from the Google Portal but was downloaded from the 2024 recruitment cycle.

The remaining job descriptions used to generate results for resume scores in Figure 1 can be found in the folder [sample_input_data/example_job_descriptions].

### LLM Tools

We used the following LLMs to perform manipulations on our input resumes:
- GPT-3.5-Turbo-0125
- GPT-4o-mini
- GPT-4o-2024-08-06 
- Claude-3.5-Sonnet
- DeepSeek-67B
- DeepSeek-V3 DeepSeek 
- Mixtral-8x7b-Instruct
- Llama3.3-70B-Instruct-Turbo

The paper details more about the sequence of LLM manipulations. 

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/heyyjudes/llm-hiring-ecosystem.git
   cd llm-hiring-ecosystem
   ```

2. **Install required libraries:**
   The main dependencies are listed in `env.yml`. Install them with:
   ```
   conda env create -f env.yml
   ```

## Experiments

- **Ski Rental**: [ski_rental.ipynb](ski-rental.ipynb) Demonstrates algorithms for the ski rental problem using calibrated machine learning predictions (Section 3 of our paper).
- **Online Scheduling**: [scheduling.ipynb](scheduling.ipynb) Implements scheduling algorithms with calibrated predictions (Section 4 of our paper).

Both notebooks rely on several other files: 
- `calibration.py` contains classes for Histogram Calibration, Bin Calibration, and Platt Scaling. 
- `model.py` contains models we use for the base predictors. 
- `ski_rental.py` contains helper functions for calculating competitive ratio.  


## Citation

If you use this code or find it helpful, please cite our paper:
```
@article{cohen2025ticketsbetteronefair,
  title={Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations},
  author={Cohen, Lee and Hsieh, Jack and Hong, Connie, and Shen, Judy},
  journal={arXiv preprint arXiv:2502.13221},
  year={2025}
}

```