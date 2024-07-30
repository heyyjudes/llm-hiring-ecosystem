import numpy as np
import yaml
import os

import pandas as pd
#need to do:  conda install conda-forge::qdrant-client
from qdrant_client import QdrantClient

#Generate word similarity score for one pair: (job description & resume). Code from: https://github.com/srbhr/Resume-Matcher/blob/main/scripts/similarity/get_score.py
def get_score(resume_string, job_description_string):
    """
    The function `get_score` uses QdrantClient to calculate the similarity score between a resume and a
    job description.

    Args:
      resume_string: The `resume_string` parameter is a string containing the text of a resume. It
    represents the content of a resume that you want to compare with a job description.
      job_description_string: The `get_score` function you provided seems to be using a QdrantClient to
    calculate the similarity score between a resume and a job description. The function takes in two
    parameters: `resume_string` and `job_description_string`, where `resume_string` is the text content
    of the resume and

    Returns:
      The function `get_score` returns the search result obtained by querying a QdrantClient with the
    job description string against the resume string provided.
    """
   # logger.info("Started getting similarity score")

    documents: List[str] = [resume_string]
    client = QdrantClient(":memory:")
    client.set_model("BAAI/bge-base-en")

    client.add(
        collection_name="demo_collection",
        documents=documents,
    )

    search_result = client.query(
        collection_name="demo_collection", query_text=job_description_string
    )
    #logger.info("Finished getting similarity score")
    similarity_score = round(search_result[0].score * 100, 2)
    
    return similarity_score 

#Wrapper to perform get_score on a list of resumes (from data_frame labeled).
def append_scores(dataframe_labeled, job_description, job_title):
    #appends score of similarity of Job title to dataframe.
    scores = [0 for i in range(dataframe_labeled)]
    counter=0
    for index, row in dataframe_labeled.iterrows():
        resumm = "".join(row.to_dict()['CV'].split('\r\n')).split('\n')
        similarity_score = get_score("".join(resumm), job_description)
        scores[counter] = similarity_score
        print(similarity_score, row['Position'], flush=True)
        counter+=1
    dataframe_labeled[job_title+" Score"] = scores
    return dataframe_labeled


if __name__ == "__main__":
    print("Scoring Data for resumes.", flush=True)
    from cleandataframe import trueLabelFunction
    df = pd.read_parquet('data/resumes.parquet', engine='pyarrow')  # raw dataframe
    # Filter the dataframe minimum cv length
    MIN_CV_LENGTH = 500
    filtered_df = df.loc[df['CV'].dropna().apply(len) >= MIN_CV_LENGTH]
    labeled_df = filtered_df.copy()
    labeled_df["True Label"] = labeled_df.apply(trueLabelFunction, axis=1)
    labeled_df = labeled_df[labeled_df["True Label"].notna()] 
    job_doordash_pm = """
    Tailor my resume to this job description and not 
    make anything up: Product Manager (Multiple Levels) - 
    Doordash: About the Team At DoorDash, we're redefining the future of on-demand delivery. 
    To do this, we're building a world-class product organization, in which each of our product managers play a critical role in helping to define and execute our vision to connect local delivery networks in cities all across the world! About The Role Product Managers at DoorDash require a sharp consumer-first eye, platform thinking and strong cross-functional collaboration. As a Product Manager at DoorDash, you will own the product strategy and vision, define the product roadmap and alignment, and help drive the execution. You will be working on mission-critical products that shape the direction of the company. You will report into one of the following pillars: Merchant, Consumer, Operational Excellence, Ads, Logistics, or New Verticals. This role is a hybrid of remote work and in-person collaboration. You’re Excited About This Opportunity Because You Will… Drive the product definition, strategy, and long term vision. You own the roadmap Work closely with cross-functional teams of designers, operators, data scientists and engineers Communicate product plans, benefits and results to key stakeholders including leadership team We’re Excited About You Because… You have 5+ years of Product Management Industry Experience You have 4+ years of user-facing experience in industries such as eCommerce, technology or multi-sided marketplaces You have proven abilities in driving product strategy, vision, and roadmap alignment You’re an execution power-house You have experience presenting business reviews to senior executives You have empathy for the users you build for You are passionate about DoorDash and the problems we are solving for About DoorDash At DoorDash, our mission to empower local economies shapes how our team members move quickly, learn, and reiterate in order to make impactful decisions that display empathy for our range of users—from Dashers to merchant partners to consumers. We are a technology and logistics company that started with door-to-door delivery, and we are looking for team members who can help us go from a company that is known for delivering food to a company that people turn to for any and all goods. DoorDash is growing rapidly and changing constantly, which gives our team members the opportunity to share their unique perspectives, solve new challenges, and own their careers. We're committed to supporting employees’ happiness, healthiness, and overall well-being by providing comprehensive benefits and perks including premium healthcare, wellness expense reimbursement, paid parental leave and more. Our Commitment to Diversity and Inclusion We’re committed to growing and empowering a more inclusive community within our company, industry, and cities. That’s why we hire and cultivate diverse teams of people from all backgrounds, experiences, and perspectives. We believe that true innovation happens when everyone has room at the table and the tools, resources, and opportunity to excel. Statement of Non-Discrimination: In keeping with our beliefs and goals, no employee or applicant will face discrimination or harassment based on: race, color, ancestry, national origin, religion, age, gender, marital/domestic partner status, sexual orientation, gender identity or expression, disability status, or veteran status. Above and beyond discrimination and harassment based on 'protected categories,' we also strive to prevent other subtler forms of inappropriate behavior (i.e., stereotyping) from ever gaining a foothold in our office. Whether blatant or hidden, barriers to success have no place at DoorDash. We value a diverse workforce – people who identify as women, non-binary or gender non-conforming, LGBTQIA+, American Indian or Native Alaskan, Black or African American, Hispanic or Latinx, Native Hawaiian or Other Pacific Islander, differently-abled, caretakers and parents, and veterans are strongly encouraged to apply. Thank you to the Level Playing Field Institute for this statement of non-discrimination. Pursuant to the San Francisco Fair Chance Ordinance, Los Angeles Fair Chance Initiative for Hiring Ordinance, and any other state or local hiring regulations, we will consider for employment any qualified applicant, including those with arrest and conviction records, in a manner consistent with the applicable regulation. If you need any accommodations, please inform your recruiting contact upon initial connection."
    """
    append_scores(labeled_df, job_doordash_pm, "DoorDash PM")
    labeled_df.to_csv("Scored_Resumes.csv")