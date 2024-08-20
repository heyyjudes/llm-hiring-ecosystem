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
# NEGATIVE_POSITION = {"QA Engineer"}   # "Java Developer"
# NEGATIVE_KEYWORD = {"QA"}             # "Java"
NEGATIVE_POSITIONS = {"UI/UX Designer", "UX/UI Designer"}
NEGATIVE_KEYWORDS = {"Design"}

'''
==================================================
Related to resume generation
==================================================
'''
# MODEL_NAME = "GPT-4o-mini"
MODEL_NAME = "Together"

import yaml
with open('llm_api_keys.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Together callable
from together import Together
from openai import OpenAI

openai_api_key = config['services']['openai']['api_key'] 
together_api_key = config['services']['together']['api_key'] 


# CALLABLES: takes in a single prompt, returns a single output
from typing import Callable
Prompt = str
ModelRequestCallable = Callable[[Prompt], str]

def together_callable(prompt: str) -> str:
    client = Together(api_key=together_api_key) 
    response = client.chat.completions.create(
            model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages = [{"role": "user", "content": prompt}],
    )

    output = response.choices[0].message.content
    return output

def gpt4o_callable(prompt: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        # model="gpt-4o",
        messages = [{"role": "user", "content": prompt}],
        )
    output = response.choices[0].message.content
    return output
from typing import List, Dict
def gpt4omini_callable(prompt: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
        )
    output = response.choices[0].message.content
    return output

MODEL_NAME_TO_CALLABLE: Dict[str, ModelRequestCallable] = {
    "Together": together_callable,
    "GPT-4o": gpt4o_callable,
    "GPT-4o-mini": gpt4omini_callable
}

# CONVERSATIONS: takes in a series of prompts, returns a single output
Role = str
Content = str
Message = Dict[Role, Content]
ModelRequestConversation = Callable[[List[Message]], str]

from typing import List, Dict
def gpt4omini_conversation(messages: List[Dict[str, str]]) -> str:
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages
    )
    output = response.choices[0].message.content
    return output
      
def gpt4o_conversation(messages: List[Dict[str, str]]) -> str:
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages
    )
    output = response.choices[0].message.content
    return output

# Dictionary that associates model names to conversation
MODEL_NAME_TO_CONVERSATION: Dict[str, ModelRequestConversation] = {
    "GPT-4o-mini Conversation": gpt4omini_conversation,
    "GPT-4o Conversation": gpt4o_conversation
}

'''
==================================================
Related to job generation and scores
==================================================
'''
# Possible choices for job names and descriptions
BITS_ORCHESTRA_PM_JOB_NAME = "Bits Orchestra PM"
BITS_ORCHESTRA_PM_JOB_DESCRIPTION = "A commitment to collaborative problem solving, agile thinking, and adaptability is essential. We are looking for a candidate who is able to balance a fast moving and changing environment with the ability to identify, investigate, and predict project risks Recruiting stages: HR interview, Tech interview **Core Responsibilities:** - Manage the full project life cycle including requirements gathering, creation of project plans and schedules, obtaining and managing resources, and facilitating project execution, deployment, and closure. - In cooperation with Technical Leads create and maintain comprehensive project documentation. - Manage Client expectations, monitor and increase CSAT level; - Plan, perform and implement process improvement initiatives. - Organize, lead, and facilitate cross-functional project teams. - Prepare weekly and monthly project status reports **What you need to Succeed:** - 1+ Year of dedicated Project Management in a production environment - Excellent organization and communication skills and the ability to communicate effectively with customers and co-workers. - Strong understanding of a Project Management Methodology (SDLC, Agile, Waterfall, etc.) - Creative mind with the ability to think outside-of-the-box. - The ability to manage multiple projects simultaneously - Experience with Jira or similar project management tool - Upper-intermediate level of English is a must"

LINKEDIN_PM_JOB_NAME = "LinkedIn Product Manager"
LINKEDIN_PM_JOB_DESCRIPTION = '''
 The driving force behind our business growth is a skilled and dedicated project management team. We’re searching for a highly qualified project manager to help us maintain our position as an innovative authority. The ideal candidate will have production experience and strong skills in developing and overseeing work plans. The project manager will also prepare and present updates regularly to relevant management channels, ensuring that our goal of innovation is being achieved.
Objectives of this role
Build and develop the project team to ensure maximum performance, by providing purpose, direction, and motivation
Lead projects from requirements definition through deployment, identifying schedules, scopes, budget estimations, and implementation plans, including risk mitigation
Coordinate internal and external resources to ensure that projects adhere to scope, schedule, and budget
Analyze project status and, when necessary, revise the scope, schedule, or budget to ensure that project requirements can be met
Establish and maintain relationships with relevant client stakeholders, providing day-to-day contact on project status and changes
Responsibilities
Establish and maintain processes for managing scope during the project lifecycle, setting quality and performance standards and assessing risks
Structure and manage integrated, multitrack performance databases for digital, print, social, broadcast, and experiential projects
Develop and maintain partnerships with third-party resources, including vendors and researchers
Assign and monitor resources to ensure project efficiency and maximize deliverables
Report project outcomes and/or risks to the appropriate management channels and escalate issues, as necessary, according to project work plan
Required skills and qualifications
Four or more years of project management experience
Experience in developing web technologies and software platforms for maximum usability
Strong attention to deadlines and budgetary guidelines
Proven success working with all levels of management
Strong written and verbal communication skills
Excellent presentation skills
Preferred skills and qualifications
Experience in developing platforms for internal processes
Experience in coaching project team members to strengthen their abilities and skill sets
'''

# The chosen job and descriptions
JOB_NAME        = LINKEDIN_PM_JOB_NAME
JOB_DESCRIPTION = LINKEDIN_PM_JOB_DESCRIPTION

# Given a model name and the job name, return a standard-format name for the CV type
def tailored_CV_name(model_name: str, job_name: str) -> str:
    return f"{model_name}-Improved {job_name} CV"
    # return f"{model_name}-Improved General PM CV"

TAILORED_CV_NAME = tailored_CV_name(model_name = MODEL_NAME, job_name = JOB_NAME)