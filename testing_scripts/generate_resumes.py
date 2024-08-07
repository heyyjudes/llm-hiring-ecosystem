import numpy as np
import yaml
import re
import pandas as pd
import seaborn as sns
from typing import Callable
from together import Together

# %load_ext autoreload
# %autoreload 2

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import sklearn.metrics as metrics

# TODO: do all the modified resumes reach the threshold for classification as hire/interview? 
#Modify Java Resumes - see if accepted

with open('llm_api_keys.yaml', 'r') as file:
    config = yaml.safe_load(file)

together_api_key = config['services']['together']['api_key'] 

# A general function type for calling a model on a string prompt
ModelRequestCallable = Callable[[str], str]

# An example of a model request callable function, for the Together API key
def request_from_Together(prompt: str) -> str:
    client = Together(api_key=together_api_key) 
    response = client.chat.completions.create(
            model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages = [{"role": "user", "content": prompt}],
    )

    output = response.choices[0].message.content
    return output

def tailor_resume(input_resume: str, job_description: str, model_name: str) -> str:
    '''
    This function accepts an input resume, a job description, and a model name (since different models may require different prompts)
    The function uses the model to tailor the resume toward the job description
        The function fails an assertion if the inputed model is not one of the listed models
    '''
    
    MODEL_NAME_LIST: list[str] = ["Together"]

    assert model_name in MODEL_NAME_LIST, f"Error: model_name ({model_name}) must be in {MODEL_NAME_LIST}"

    if model_name == "Together":

        # Design prompt
        
        prompt: str = f"Tailor my resume to this job description and do not make anything up. It is imperative that you provide only the content of the CV without any additional explanations, notes, or comments."
        prompt += f" This is the job description: {job_description}"
        prompt += f" This is my resume: {input_resume}"

        # Ask the model
        model_request_callable: ModelRequestCallable = request_from_Together
        output: str = model_request_callable(prompt)

        return output

    raise Exception("This should never be reached: make sure MODEL_NAME_LIST contains only supported models")

    # client = Together(api_key=together_api_key) 
    # job_description: str = "Product Manager (Multiple Levels) - Doordash: About the Team At DoorDash, we're redefining the future of on-demand delivery. To do this, we're building a world-class product organization, in which each of our product managers play a critical role in helping to define and execute our vision to connect local delivery networks in cities all across the world! About The Role Product Managers at DoorDash require a sharp consumer-first eye, platform thinking and strong cross-functional collaboration. As a Product Manager at DoorDash, you will own the product strategy and vision, define the product roadmap and alignment, and help drive the execution. You will be working on mission-critical products that shape the direction of the company. You will report into one of the following pillars: Merchant, Consumer, Operational Excellence, Ads, Logistics, or New Verticals. This role is a hybrid of remote work and in-person collaboration. You’re Excited About This Opportunity Because You Will… Drive the product definition, strategy, and long term vision. You own the roadmap Work closely with cross-functional teams of designers, operators, data scientists and engineers Communicate product plans, benefits and results to key stakeholders including leadership team We’re Excited About You Because… You have 5+ years of Product Management Industry Experience You have 4+ years of user-facing experience in industries such as eCommerce, technology or multi-sided marketplaces You have proven abilities in driving product strategy, vision, and roadmap alignment You’re an execution power-house You have experience presenting business reviews to senior executives You have empathy for the users you build for You are passionate about DoorDash and the problems we are solving for About DoorDash At DoorDash, our mission to empower local economies shapes how our team members move quickly, learn, and reiterate in order to make impactful decisions that display empathy for our range of users—from Dashers to merchant partners to consumers. We are a technology and logistics company that started with door-to-door delivery, and we are looking for team members who can help us go from a company that is known for delivering food to a company that people turn to for any and all goods. DoorDash is growing rapidly and changing constantly, which gives our team members the opportunity to share their unique perspectives, solve new challenges, and own their careers. We're committed to supporting employees’ happiness, healthiness, and overall well-being by providing comprehensive benefits and perks including premium healthcare, wellness expense reimbursement, paid parental leave and more. Our Commitment to Diversity and Inclusion We’re committed to growing and empowering a more inclusive community within our company, industry, and cities. That’s why we hire and cultivate diverse teams of people f"+"rom all backgrounds, experiences, and perspectives. We believe that true innovation happens when everyone has room at the table and the tools, resources, and opportunity to excel. Statement of Non-Discrimination: In keeping with our beliefs and goals, no employee or applicant will face discrimination or harassment based on: race, color, ancestry, national origin, religion, age, gender, marital/domestic partner status, sexual orientation, gender identity or expression, disability status, or veteran status. Above and beyond discrimination and harassment based on 'protected categories,' we also strive to prevent other subtler forms of inappropriate behavior (i.e., stereotyping) from ever gaining a foothold in our office. Whether blatant or hidden, barriers to success have no place at DoorDash. We value a diverse workforce – people who identify as women, non-binary or gender non-conforming, LGBTQIA+, American Indian or Native Alaskan, Black or African American, Hispanic or Latinx, Native Hawaiian or Other Pacific Islander, differently-abled, caretakers and parents, and veterans are strongly encouraged to apply. Thank you to the Level Playing Field Institute for this statement of non-discrimination. Pursuant to the San Francisco Fair Chance Ordinance, Los Angeles Fair Chance Initiative for Hiring Ordinance, and any other state or local hiring regulations, we will consider for employment any qualified applicant, including those with arrest and conviction records, in a manner consistent with the applicable regulation. If you need any accommodations, please inform your recruiting contact upon initial connection."
    
    # response = client.chat.completions.create(
    #         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #         messages=[{"role": "user", "content": "Tailor my resume to this job description and not make anything up:" 
    #                    + job_description + "and this is my resume:" + input}],
    #                    #Modify the following resume to help me get a "+ position+" Job:" + resume}],
    #     )

    #     #Trim out the AI conversation (I hope this faldskraewr! stuff).
    # output = response.choices[0].message.content
    # output = output[output.find("\n"):output.rfind('\n')]
    # output = "".join("".join(output.split('\r\n')).split('\n'))
    # return output

def create_modified_resumes(dataframe, num_of_resumes, newposition):
         # example usage of Together AI 
     all_modified_resumes = [0 for i in range(0, len(dataframe))]
     counter = 0
     for index, row in dataframe.iterrows():
         resumm = "".join(row.to_dict()['CV'].split('\r\n')).split('\n')
         all_modified_resumes[counter] = modify_resume("".join(resumm), newposition)
         print(index, flush=True)
         counter+=1
         if counter > num_of_resumes:
             break

#     #Returns dataframe with extra column with extra modified resumes.
     dataframe[newposition+" Modified CV"] = all_modified_resumes
     return dataframe

if __name__ == "__main__":
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
    