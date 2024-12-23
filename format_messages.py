from typing import List, Dict, Union, Optional, Any

'''
Module that formats input messages to LLM api calls from user inputs of:
1. Original CV
2. Job Description to Tailor To (Optional)
3. Custom Tailoring Prompt (Optional)
'''

def get_prompt_messages_general_conversation(original_cv: str)-> List[Dict[str, str]]:
        '''
        This function outputs the desired prompt, in conversation style. It follows our original results.
        ''' 

        message1A = f"Can you improve my resume?"
        message1A += f'''I will be punished if I lie about my qualifications, 
        so it is imperative that the new resume does not add any facts that are not in the original resume.'''
        message1A += f"Also, please do not include any additional notes or explanation: I just want the text of the new resume. Do not even write 'here is the resume', I just need the plain text."
        message1A += f"Again, ensure that the output accurately represents my actual responsibilities, experiences, and skills."
        message1A += f"This is my resume: {original_cv}"
        messages = [{"role": "user", "content": message1A}]

        return messages
    
def get_prompt_messages_general_conversation_with_job_description(original_cv: str, job_description: str)-> List[Dict[str, str]]:
    if len(job_description)==0:
        raise Exception('Job Description is empty for usage of prompt requiring description.')
        return 
    message1A = f"I'm applying for a project manager role. This is the job description: {job_description}.\nPlease wait for further instructions."
    message1B = f"Got it! I'll wait for your next instructions. Just let me know how you'd like to proceed or if you need help with anything specific regarding your application."
    message2AA = f"Also, please do not include any additional notes or explanation: I just want the text of the new resume. Do not even write 'here is the resume', I just need the plain text."
    message2A = f''' Can you revise my resume for this job? I will be disqualified if I 
    lie about my qualifications, so it is imperative that the new resume accurately 
    represents my actual responsibilities and experiences from my original resume. 
    Also, ensure that the output accurately represents my actual responsibilities and experiences. 
    Don't add things not aren't applicable. Also, please do not include any additional notes or explanation: 
    I just want the text of the new resume. This is my resume: {original_cv}'''
    message3A = 'Can you remove the information that is false, or incomplete?'
    messages = [
        {"role": "user", "content": message1A},
        {"role": "assistant", "content": message1B},
        {"role": "user", "content": message2AA},
        {"role": "user", "content": message2A},
        {"role": "user", "content": message3A}
    ]
    return messages

def get_prompt_messages_custom_prompt(original_cv: str, custom_prompt: str, job_description):
    if len(custom_prompt)==0:
        raise Exception('Prompt is empty for usage of prompt requiring custom prompt.')
        return
    
    message1A= custom_prompt
    if len(job_description)>0:
        message1A+=f"This is the description of the job I am applying for: {job_description}. Please wait for further instructions.\n"
    message1A+=f"This is my resume: {original_cv}."
    messages = [{"role": "user", "content": message1A}]
    return messages

def generate_messages_per_resume(prompt_type, input_cv, job_description: str='', custom_prompt: str=''):
    if prompt_type == 'General Conversation w/o Job Description':
        return get_prompt_messages_general_conversation(original_cv = input_cv)
    elif prompt_type == 'General Conversation w/ Job Description':
        return get_prompt_messages_general_conversation_with_job_description(original_cv = input_cv, job_description=job_description)
    elif prompt_type == 'Custom':
        if len(custom_prompt)==0:
            raise Exception('Custom Prompt Description is empty for usage of prompt of type: '+prompt_type+".")
            return 
        return get_prompt_messages_custom_prompt(original_cv = input_cv, job_description = job_description, custom_prompt=custom_prompt)