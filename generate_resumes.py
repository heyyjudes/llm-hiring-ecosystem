import numpy as np
import yaml

import pandas as pd
import seaborn as sns
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



def create_modified_resumes(unmodifiedresumes, num_of_resumes, originalposition):
    # example usage of Together AI 
    counter_loop = 0
    modifiedresumes = []
    for resume in unmodifiedresumes[0:num_of_resumes]:
        client = Together(api_key=together_api_key) 
        position = "Project Manager"
        job_descrip = "Tailor my resume to this job description and not make anything up: Product Manager (Multiple Levels) - Doordash: About the Team At DoorDash, we're redefining the future of on-demand delivery. To do this, we're building a world-class product organization, in which each of our product managers play a critical role in helping to define and execute our vision to connect local delivery networks in cities all across the world! About The Role Product Managers at DoorDash require a sharp consumer-first eye, platform thinking and strong cross-functional collaboration. As a Product Manager at DoorDash, you will own the product strategy and vision, define the product roadmap and alignment, and help drive the execution. You will be working on mission-critical products that shape the direction of the company. You will report into one of the following pillars: Merchant, Consumer, Operational Excellence, Ads, Logistics, or New Verticals. This role is a hybrid of remote work and in-person collaboration. You’re Excited About This Opportunity Because You Will… Drive the product definition, strategy, and long term vision. You own the roadmap Work closely with cross-functional teams of designers, operators, data scientists and engineers Communicate product plans, benefits and results to key stakeholders including leadership team We’re Excited About You Because… You have 5+ years of Product Management Industry Experience You have 4+ years of user-facing experience in industries such as eCommerce, technology or multi-sided marketplaces You have proven abilities in driving product strategy, vision, and roadmap alignment You’re an execution power-house You have experience presenting business reviews to senior executives You have empathy for the users you build for You are passionate about DoorDash and the problems we are solving for About DoorDash At DoorDash, our mission to empower local economies shapes how our team members move quickly, learn, and reiterate in order to make impactful decisions that display empathy for our range of users—from Dashers to merchant partners to consumers. We are a technology and logistics company that started with door-to-door delivery, and we are looking for team members who can help us go from a company that is known for delivering food to a company that people turn to for any and all goods. DoorDash is growing rapidly and changing constantly, which gives our team members the opportunity to share their unique perspectives, solve new challenges, and own their careers. We're committed to supporting employees’ happiness, healthiness, and overall well-being by providing comprehensive benefits and perks including premium healthcare, wellness expense reimbursement, paid parental leave and more. Our Commitment to Diversity and Inclusion We’re committed to growing and empowering a more inclusive community within our company, industry, and cities. That’s why we hire and cultivate diverse teams of people f"+"rom all backgrounds, experiences, and perspectives. We believe that true innovation happens when everyone has room at the table and the tools, resources, and opportunity to excel. Statement of Non-Discrimination: In keeping with our beliefs and goals, no employee or applicant will face discrimination or harassment based on: race, color, ancestry, national origin, religion, age, gender, marital/domestic partner status, sexual orientation, gender identity or expression, disability status, or veteran status. Above and beyond discrimination and harassment based on 'protected categories,' we also strive to prevent other subtler forms of inappropriate behavior (i.e., stereotyping) from ever gaining a foothold in our office. Whether blatant or hidden, barriers to success have no place at DoorDash. We value a diverse workforce – people who identify as women, non-binary or gender non-conforming, LGBTQIA+, American Indian or Native Alaskan, Black or African American, Hispanic or Latinx, Native Hawaiian or Other Pacific Islander, differently-abled, caretakers and parents, and veterans are strongly encouraged to apply. Thank you to the Level Playing Field Institute for this statement of non-discrimination. Pursuant to the San Francisco Fair Chance Ordinance, Los Angeles Fair Chance Initiative for Hiring Ordinance, and any other state or local hiring regulations, we will consider for employment any qualified applicant, including those with arrest and conviction records, in a manner consistent with the applicable regulation. If you need any accommodations, please inform your recruiting contact upon initial connection."
        
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": job_descrip + "This is my resume:" + resume}],
                       #Modify the following resume to help me get a "+ position+" Job:" + resume}],
        )

        #Trim out the AI conversation (I hope this faldskraewr! stuff).
        output = response.choices[0].message.content
        output = output[output.find("\n"):output.rfind('\n')]
        output = "".join("".join(output.split('\r\n')).split('\n'))
        modifiedresumes.append(output)
        print("Resume: %d" % counter_loop)
        counter_loop+=1
    with open(originalposition+'to'+position+'.txt', 'w') as f:
        for resume in modifiedresumes:
            f.write(f"{resume}\n")
    return modifiedresumes

java_to_pm_mod_resume = create_modified_resumes(list(java_dev_occupation_df['CV']), len(java_dev_occupation_df), "JavaDeveloper")
