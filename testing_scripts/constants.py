'''
This file contains constants that are used across different sections
This allow us to run later sections without having to run all previous sections
'''

'''
==================================================
Related to true labels
==================================================
'''
POSITIVE_LABEL, NEGATIVE_LABEL = 1, 0       # Classification labels

# Keywords for determining true labels
POSITIVE_POSITIONS = {"Project Manager"}
POSITIVE_KEYWORDS = {"Project Manager"}
NEGATIVE_POSITION = {"QA Engineer"}   # "Java Developer"
NEGATIVE_KEYWORD = {"QA"}             # "Java"
# NEGATIVE_POSITIONS = {"UI/UX Designer", "UX/UI Designer"}
# NEGATIVE_KEYWORDS = {"Design"}

'''
==================================================
Related to resume generation
==================================================
'''
MODEL_NAME = "Together"

from typing import Callable
ModelRequestCallable = Callable[[str], str]     # Takes in a prompt string, outputs a generated string

import yaml
with open('llm_api_keys.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Together callable
from together import Together
together_api_key = config['services']['together']['api_key'] 

def together_callable(prompt: str) -> str:
    client = Together(api_key=together_api_key) 
    response = client.chat.completions.create(
            model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages = [{"role": "user", "content": prompt}],
    )

    output = response.choices[0].message.content
    return output

# Dictionary that associates model names to callables
MODEL_NAME_TO_CALLABLE = {
    "Together" : together_callable
}


'''
==================================================
Related to job generation and scores
==================================================
'''
# Possible choices for job names and descriptions
BITS_ORCHESTRA_PM_JOB_NAME = "Bits Orchestra PM"
BITS_ORCHESTRA_PM_JOB_DESCRIPTION = "A commitment to collaborative problem solving, agile thinking, and adaptability is essential. We are looking for a candidate who is able to balance a fast moving and changing environment with the ability to identify, investigate, and predict project risks Recruiting stages: HR interview, Tech interview **Core Responsibilities:** - Manage the full project life cycle including requirements gathering, creation of project plans and schedules, obtaining and managing resources, and facilitating project execution, deployment, and closure. - In cooperation with Technical Leads create and maintain comprehensive project documentation. - Manage Client expectations, monitor and increase CSAT level; - Plan, perform and implement process improvement initiatives. - Organize, lead, and facilitate cross-functional project teams. - Prepare weekly and monthly project status reports **What you need to Succeed:** - 1+ Year of dedicated Project Management in a production environment - Excellent organization and communication skills and the ability to communicate effectively with customers and co-workers. - Strong understanding of a Project Management Methodology (SDLC, Agile, Waterfall, etc.) - Creative mind with the ability to think outside-of-the-box. - The ability to manage multiple projects simultaneously - Experience with Jira or similar project management tool - Upper-intermediate level of English is a must"
DOORDASH_PM_JOB_NAME = "Doordash Product Manager"
DOORDASH_PM_JOB_DESCRIPTION = "Product Manager (Multiple Levels) - Doordash: About the Team At DoorDash, we're redefining the future of on-demand delivery. To do this, we're building a world-class product organization, in which each of our product managers play a critical role in helping to define and execute our vision to connect local delivery networks in cities all across the world! About The Role Product Managers at DoorDash require a sharp consumer-first eye, platform thinking and strong cross-functional collaboration. As a Product Manager at DoorDash, you will own the product strategy and vision, define the product roadmap and alignment, and help drive the execution. You will be working on mission-critical products that shape the direction of the company. You will report into one of the following pillars: Merchant, Consumer, Operational Excellence, Ads, Logistics, or New Verticals. This role is a hybrid of remote work and in-person collaboration. You’re Excited About This Opportunity Because You Will… Drive the product definition, strategy, and long term vision. You own the roadmap Work closely with cross-functional teams of designers, operators, data scientists and engineers Communicate product plans, benefits and results to key stakeholders including leadership team We’re Excited About You Because… You have 5+ years of Product Management Industry Experience You have 4+ years of user-facing experience in industries such as eCommerce, technology or multi-sided marketplaces You have proven abilities in driving product strategy, vision, and roadmap alignment You’re an execution power-house You have experience presenting business reviews to senior executives You have empathy for the users you build for You are passionate about DoorDash and the problems we are solving for About DoorDash At DoorDash, our mission to empower local economies shapes how our team members move quickly, learn, and reiterate in order to make impactful decisions that display empathy for our range of users—from Dashers to merchant partners to consumers. We are a technology and logistics company that started with door-to-door delivery, and we are looking for team members who can help us go from a company that is known for delivering food to a company that people turn to for any and all goods. DoorDash is growing rapidly and changing constantly, which gives our team members the opportunity to share their unique perspectives, solve new challenges, and own their careers. We're committed to supporting employees’ happiness, healthiness, and overall well-being by providing comprehensive benefits and perks including premium healthcare, wellness expense reimbursement, paid parental leave and more. Our Commitment to Diversity and Inclusion We’re committed to growing and empowering a more inclusive community within our company, industry, and cities. That’s why we hire and cultivate diverse teams of people f"+"rom all backgrounds, experiences, and perspectives. We believe that true innovation happens when everyone has room at the table and the tools, resources, and opportunity to excel. Statement of Non-Discrimination: In keeping with our beliefs and goals, no employee or applicant will face discrimination or harassment based on: race, color, ancestry, national origin, religion, age, gender, marital/domestic partner status, sexual orientation, gender identity or expression, disability status, or veteran status. Above and beyond discrimination and harassment based on 'protected categories,' we also strive to prevent other subtler forms of inappropriate behavior (i.e., stereotyping) from ever gaining a foothold in our office. Whether blatant or hidden, barriers to success have no place at DoorDash. We value a diverse workforce – people who identify as women, non-binary or gender non-conforming, LGBTQIA+, American Indian or Native Alaskan, Black or African American, Hispanic or Latinx, Native Hawaiian or Other Pacific Islander, differently-abled, caretakers and parents, and veterans are strongly encouraged to apply. Thank you to the Level Playing Field Institute for this statement of non-discrimination. Pursuant to the San Francisco Fair Chance Ordinance, Los Angeles Fair Chance Initiative for Hiring Ordinance, and any other state or local hiring regulations, we will consider for employment any qualified applicant, including those with arrest and conviction records, in a manner consistent with the applicable regulation. If you need any accommodations, please inform your recruiting contact upon initial connection."

# The chosen job and descriptions
JOB_NAME        = BITS_ORCHESTRA_PM_JOB_NAME
JOB_DESCRIPTION = BITS_ORCHESTRA_PM_JOB_DESCRIPTION

# Given a model name and the job name, return a standard-format name for the CV type
def tailored_CV_name(model_name: str, job_name: str) -> str:
    return f"{model_name}-Improved {job_name} CV"

TAILORED_CV_NAME = tailored_CV_name(model_name = MODEL_NAME, job_name = JOB_NAME)