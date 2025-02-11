"""
This module provides functions to improve resumes/CVs using various LLM APIs.

Running modify_cv takes in the following inputs and outputs the modified CVs in a .csv file. Optional inputs also have default values in the code:
1. Input CVs (Filepath(s), Required)
2. Output Directory (Filepath, Required)
3. Prompt Template (Filepath (.json or .txt), Required) - Prompt with placeholder for {original_cv} (optional placeholder for {job_description}).
4. Job Description for Prompt (.txt, Optional) - User-inputted prompt for the last prompt type.
5. LLM Provider (String - 1 of OpenAI, Together, Anthropic., Required)
6. API-Key (Filepath, Required - path to api_keys.yaml file).
7. Model (String, Optional) - name of model to use (besides default).

It outputs a csv, timestamped, with one column corresponding to the modified resume/CV text. 

Example Usage:
python3 modify_cv.py test_cvs.csv test_folder --prompt-template test_template.txt --prompt-job-description scalable_job_description.txt --provider openai --api-key llm_api_keys.yaml 
python3 modify_cv.py jan_samples.csv jan_folder --prompt-template jan_prompt.txt --provider openai --model gpt-3.5-turbo --api-key llm_api_keys.yaml 
  
Example Input Files can be found in sample_input_data.
"""
#Change generate_resume_messages -> format files that this code can take as an input. 
# or somehow format_file, prompt_content as input, and 
# internal function here that combines format_file & prompt_content tgt as one. 
import time
import argparse
import yaml
import pandas as pd
from typing import List
from pathlib import Path
from enum import Enum
import logging
import json
from datetime import datetime
from together import Together
import os

from langchain_core.prompts import PromptTemplate

from openai import OpenAI
from anthropic import Anthropic

from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

time_sleep_anthropic = 20
time_sleep_openai=20
MAX_TOKENS=1024
openai_individual_batch_threshold=5

#Set up Prompts Code
def read_template(template_path):
    template_path = str(template_path)
    if template_path.endswith(".json"):
        template_json = json.load(open(template_path, 'r'))
        final_template = list(template_json.values())
    else:
        str_template = open(template_path, 'r').read()
        final_template = PromptTemplate.from_template(template = str_template)
    return final_template
    
def format_message_json(prompt_template:List, input_cv, job_desc:str = ''):
    messages_formatted = [{'role':i['role'], 'content':i['content'].replace("{original_cv}", input_cv).replace("{job_description}", job_desc)} for i in prompt_template]
    return messages_formatted

def format_message_txt(prompt_template, input_cv, job_desc:str= ''):
    messages_string = prompt_template.format(original_cv = input_cv, job_description=job_desc)
    messages_formatted = [{"role": "user", "content": messages_string}]
    return messages_formatted

class AnthropicClient:
    """Anthropic-specific implementation"""
    def __init__(self, input_api_key:str, model: str, prompt_template_path:str, prompt_template:str, prompt_job_desc_filename:str, prompt_job_description: str):
       #Initialize API client with api keys & model: also stores information about tailoring resume prompt for this run of the program.

        self.client = Anthropic(api_key=input_api_key)
        self.model = model if model else "claude-3-sonnet-20240229"
        self.prompt_template_path = prompt_template_path
        self.prompt_template = prompt_template
        self.prompt_job_description_filename = prompt_job_desc_filename if prompt_job_desc_filename else ''
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''

        return 
    
    def format_messages(self, input_cv:str):
        if self.prompt_template_path.endswith('.json'):
            return format_message_json(self.prompt_template, input_cv, self.prompt_job_description)
        return format_message_txt(prompt_template=self.prompt_template, input_cv=input_cv, job_desc=self.prompt_job_description)
    

    def __client_api_call_function_request_input_cv(self, input_cv:str, id: int)->Request:
        #Private Method: from the inputs of an input-cv, formats it into request that can be passed into the OpenAI API.
        messages = self.format_messages(input_cv = input_cv)
        return_request = Request(
            custom_id=str(id),
            params=MessageCreateParamsNonStreaming(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages = messages
            )
        )
        return return_request

    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame) -> List[str]:
        #Public Method - generates modified CVs from dataframe of inputted CVs.

        if len(cv_s_dataframe.columns)>1:
            raise Exception("More than one column of resumes inputted. Please reformat input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}" 

        original_cv_s = list(cv_s_dataframe[to_be_modified_col])

        #Formats all input CVs in a list of Request objects.
        cv_batch_requests = [self.__client_api_call_function_request_input_cv(input_cv=original_cv_s[i], id=i) for i in range(len(original_cv_s))]

        #Creates a Batch object from these CV Requests. 
        cv_batch_requests_output = self.client.beta.messages.batches.create(requests=cv_batch_requests)
        batch_id = cv_batch_requests_output.id

        #Send Batch Object to LLM API for generation.
        while True:
            message_batch =self.client.beta.messages.batches.retrieve(batch_id)
            if message_batch.processing_status =='ended':
                break
            print(f"Batch {message_batch.id} is still processing...")
            time.sleep(time_sleep_anthropic)

        #Messages Batch Processing is done.
        output_resumes = []

        #Filter results by succeeded (if so, append to results), otherwise, add a placeholder of not_succeeded.
        for result in self.client.beta.messages.batches.results(message_batch.id):
            if result.result.type == 'succeeded':
                output_resumes.append(result.result.message.content[0].text)
            else:
                match result.result.type:
                    case "errored":
                        if result.result.error.type == "invalid_request":
                            # Request body must be fixed before re-sending request
                            print(f"Validation error {result.custom_id}")
                        else:
                            # Request can be retried directly
                            print(f"Server error {result.custom_id}")
                    case "expired":
                        print(f"Request expired {result.custom_id}")
                output_resumes.append("not_succeeded")

        #Save outputted results to a dataframe.
        cv_s_dataframe[modified_col_name] = output_resumes
        return cv_s_dataframe[[modified_col_name]]


class DeepSeekClient:
    """DeepSeek specific implementation:
    Everything is similar except for the base_url: "https://api.deepseek.com"
    """

    def __init__(self, input_api_key: str, model: str, prompt_template_path: str, prompt_template: str,
                 prompt_job_desc_filename: str, prompt_job_description: str):
        # Initialize API client with api keys & model: also stores information about tailoring resume prompt for this run of the program.

        self.client = OpenAI(api_key=input_api_key, base_url="https://api.deepseek.com")
        self.model = model if model else 'deepseek-chat'
        self.prompt_template_path = prompt_template_path
        self.prompt_template = prompt_template
        self.prompt_job_description_filename = prompt_job_desc_filename if prompt_job_desc_filename else ''
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''

        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        self.output_file_name = f"deepseek-{current_datetime_str}.jsonl"

        self.time_marker = current_datetime_str
        self.num_generated = 0

        return

    def format_messages(self, input_cv: str):
        if self.prompt_template_path.endswith('.json'):
            return format_message_json(self.prompt_template, input_cv, self.prompt_job_description)
        return format_message_txt(prompt_template=self.prompt_template, input_cv=input_cv,
                                  job_desc=self.prompt_job_description)

    def __client_api_call_function(self, messages)->str:
        #Private Method: from the inputs of an message, formats it into request that can be passed into the OpenAI API.
        response = self.client.chat.completions.create(
            model=self.model,
            messages = messages
        )
        output = response.choices[0].message.content
        return output

    # The following functions achieve the same purpose of modifying resumes, but DO NOT use the new Batch API functionality.
    def __generate_individal_cv(self, input_cv: str) -> str:
        messages = self.format_messages(input_cv=input_cv)
        output: str = self.__client_api_call_function(messages)
        self.num_generated += 1
        print(f"Generated {self.num_generated} resume.")
        return output

    def __generate_group_of_cv_s_from_individual_calls(self, cv_s_dataframe: pd.DataFrame):
        # Generate modified column name
        if len(cv_s_dataframe.columns) > 1:
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")

        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}"

        # Generate resumes together.
        generated_cvs = []
        for i, cv in enumerate(cv_s_dataframe[to_be_modified_col]):
            try:
                generated_cvs.append(self.__generate_individal_cv(input_cv=cv))
            except:
                new_df = pd.DataFrame()
                new_df[modified_col_name] = generated_cvs
                new_df.to_csv(f"saved_generations_step_{i}.csv")

        cv_s_dataframe[modified_col_name] = generated_cvs
        return cv_s_dataframe[[modified_col_name]]

    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame):
        # For deepseek: generate CVs from individual calls
        return self.__generate_group_of_cv_s_from_individual_calls(cv_s_dataframe)

class OpenAIClient:
    """OpenAI-specific implementation"""
    def __init__(self, input_api_key:str, model: str, prompt_template_path:str, prompt_template:str, prompt_job_desc_filename:str, prompt_job_description: str):
        #Initialize API client with api keys & model: also stores information about tailoring resume prompt for this run of the program.

        self.client = OpenAI(api_key=input_api_key)
        self.model = model if model else 'gpt-4'
        self.prompt_template_path = prompt_template_path
        self.prompt_template = prompt_template
        self.prompt_job_description_filename = prompt_job_desc_filename if prompt_job_desc_filename else ''
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''
        self.open_ai_batch_id = None

        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        self.output_file_name = f"{current_datetime_str}.jsonl"

        self.time_marker = current_datetime_str
        self.num_generated = 0
        
        return 
    
    def format_messages(self, input_cv:str):
        if self.prompt_template_path.endswith('.json'):
            return format_message_json(self.prompt_template, input_cv, self.prompt_job_description)
        return format_message_txt(prompt_template=self.prompt_template, input_cv=input_cv, job_desc=self.prompt_job_description)
       

    def __client_api_call_function(self, messages)->str:
        #Private Method: from the inputs of an message, formats it into request that can be passed into the OpenAI API.
        response = self.client.chat.completions.create(
            model=self.model,
            messages = messages
        )
        output = response.choices[0].message.content
        return output
    
    def __create_batch_file_input(self, cv_s_dataframe: pd.DataFrame, to_be_modified_col: str = 'CV'):
        #Private Method: from the inputs of an input-cv, first combines the CVs with prompt data to generate LLM API messages, then formats it into json-L file that can be passed as inputs into the OpenAI API.
        original_cv_s = list(cv_s_dataframe[to_be_modified_col])
        batch_inputs = []

        for index, row in cv_s_dataframe.iterrows():
            batch_inputs.append({"custom_id":f"resume-request-{str(index)}",
                                 'method':"POST",
                                 'url': "/v1/chat/completions",
                                 'body':{'model':self.model, 'messages':self.format_messages(input_cv = row[to_be_modified_col])}})

        formatted_inputs_file_name = "openai_formatted_inputs_"+self.time_marker+".jsonl"
        with open(formatted_inputs_file_name, "w") as f:
            for item in batch_inputs:
                f.write(json.dumps(item) + "\n")
        return formatted_inputs_file_name 
    
    def __send_group_of_cv_s_batch(self, cv_s_dataframe: pd.DataFrame, to_be_modified_col: str = 'CV'):
        #Private Method: creates input jsonL file from input cvs, and creates a corresponding batch object.

        batch_input_file_name = self.__create_batch_file_input(cv_s_dataframe=cv_s_dataframe, to_be_modified_col=to_be_modified_col)
        
        batch_input_file = self.client.files.create(file=open(batch_input_file_name, "rb"),purpose="batch")
        batch_input_file_id = batch_input_file.id

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": self.model+"_resume_generation"+"_time_"+self.time_marker
            }
        )

        os.remove(batch_input_file_name)

        self.batch_id = batch_object.id
        return batch_object
    
    def __status_ended(self):
        #Private Method to check if sent batch has ended. 

        if self.batch_id:
            if self.client.batches.retrieve(self.batch_id).status not in ['in_progress', 'validating', 'cancelling', 'finalizing']:
                print(self.client.batches.retrieve(self.batch_id).status)
                return True
            print(f"Batch {self.batch_id} is still "+ self.client.batches.retrieve(self.batch_id).status+".")
            return False
        else:
            raise Exception("Empty batch id.")
            return

    def __cancel_batch_of_cvs(self):
        #Private Method to cancel batch if needed. Not ever used in this module.
        self.batches.cancel(self.batch_id)
        self.batch_id = None
        return 
        
    def __generate_group_of_cv_s_batch(self, cv_s_dataframe: pd.DataFrame):
        #Private Method - generates modified CVs from dataframe of inputted CVs with BATCH processing.

        if len(cv_s_dataframe.columns)>1:
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]

        self.__send_group_of_cv_s_batch(cv_s_dataframe=cv_s_dataframe, to_be_modified_col=to_be_modified_col)

        if self.batch_id is None:
            raise Exception("There is no batch id - either this job has been previously cancelled or something is wrong.") 
        
        while True:
            if self.__status_ended():
                break
            time.sleep(time_sleep_openai)

        final_batch_status = self.client.batches.retrieve(self.batch_id).status
        if final_batch_status != 'completed':
            raise Exception(self.batch_id+" finished, but was: "+final_batch_status)
        
        else:
            #Ended.
            output_id = self.client.batches.retrieve(self.batch_id).output_file_id
            output_file_response = self.client.files.content(output_id)

            json_data = output_file_response.content.decode('utf-8')

            #Filter output results by if succeeded (if so, append to results), otherwise, keep the placeholder of not_succeeded.
            output_resumes = ['not_successfully_modified' for i in range(len(cv_s_dataframe))]
            # Open the specified file in write mode
            for line in json_data.splitlines():
                # Parse the JSON record (line) to validate it
                json_record = json.loads(line)
                
                current_output = ''
                # Extract and print the custom_id
                custom_id = json_record.get("custom_id")
                custom_id_no = custom_id.split("-")[-1]
                
                # Navigate to the 'choices' key within the 'response' -> 'body'
                choices = json_record.get("response", {}).get("body", {}).get("choices", [])
                
                # Loop through the choices to find messages with the 'assistant' role
                for choice in choices:
                    message = choice.get("message", {})
                    if message.get("role") == "assistant":
                        assistant_content = message.get("content")
                        current_output+=f"\n {assistant_content}\n"
                                         
                output_resumes[int(custom_id_no)] = current_output

            #Save outputted results to a dataframe of the modified resumes.
            to_be_modified_col = cv_s_dataframe.columns[-1]
            modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}" 

            cv_s_dataframe[modified_col_name] = output_resumes
            return cv_s_dataframe[[modified_col_name]]

    #The following functions achieve the same purpose of modifying resumes, but DO NOT use the new Batch API functionality. 
    def __generate_individal_cv(self, input_cv: str) -> str:
        messages = self.format_messages(input_cv = input_cv)
        output: str = self.__client_api_call_function(messages)
        self.num_generated+=1
        print(f"Generated {self.num_generated} resume.")
        return output

    def __generate_group_of_cv_s_from_individual_calls(self, cv_s_dataframe: pd.DataFrame):
        #Generate modified column name
        if len(cv_s_dataframe.columns)>1:
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}" 

        #Generate resumes together.
        generate = lambda cv : self.__generate_individal_cv(input_cv = cv)
        cv_s_dataframe[modified_col_name]= cv_s_dataframe[to_be_modified_col].apply(generate)
        return cv_s_dataframe[[modified_col_name]]
    
    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame):
        #Wrappper function that generates group of cv-s with either individual requests to LLM API (if size small enough) or batch requests to LLM API.
        if len(cv_s_dataframe)<=openai_individual_batch_threshold:
            return self.__generate_group_of_cv_s_from_individual_calls(cv_s_dataframe)
        else:
            return self.__generate_group_of_cv_s_batch(cv_s_dataframe)


class TogetherAIClient:
    """TogetherAI-specific implementation"""
    def __init__(self, input_api_key:str, model: str, prompt_template_path:str, prompt_template:str, prompt_job_desc_filename:str, prompt_job_description: str):
        #Initialize API client with api keys & model: also stores information about tailoring resume prompt for this run of the program.
        self.client = Together(api_key=input_api_key)
        self.model = model if model else "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.prompt_template_path = prompt_template_path
        self.prompt_template = prompt_template
        self.prompt_job_description_filename= prompt_job_desc_filename if prompt_job_desc_filename else ''
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''
        self.num_generated = 0

        return 
    
    def format_messages(self, input_cv:str):
        if self.prompt_template_path.endswith('.json'):
            return format_message_json(self.prompt_template, input_cv, self.prompt_job_description)
        return format_message_txt(prompt_template=self.prompt_template, input_cv=input_cv, job_desc=self.prompt_job_description)
        
    def __client_api_call_function(self, messages)->str:
        #Prepares chat completion request from input messages.
        response = self.client.chat.completions.create(
            model=self.model,
            messages = messages
        )
        output = response.choices[0].message.content
        return output

    def __generate_one_cv(self, input_cv: str) -> str:
        #Modifies a SINGULAR input cv from LLM API request. 
        messages = self.format_messages(input_cv)
        output: str = self.__client_api_call_function(messages)
        self.num_generated+=1
        print(f"Generated {self.num_generated} resume.")
        return output

    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame):
        # Iteratively class the above __generate_one_cv function on resumes in our dataframe.
        if len(cv_s_dataframe.columns)>1:
            print(cv_s_dataframe.columns)
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = f"Modified_{self.model}_of_{to_be_modified_col}_Model{self.model}"

        #Iterative calls with Lambda Function
        generate = lambda cv : self.__generate_one_cv(input_cv = cv)
        cv_s_dataframe[modified_col_name]= cv_s_dataframe[to_be_modified_col].apply(generate)

        #Saves output results.
        return cv_s_dataframe[[modified_col_name]]


#Inputs are the resumes, the Model Name, and the prompt name.
#From the prompt name, we also figure where or not there is a conversation prompt.
#There is also a functionality for a general prompt, which we will implement later.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve resumes using various LLM providers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "resumes",
        type=Path,
        nargs='+',
        help="Path to one or more resume files to improve"
    )
    parser.add_argument(
        "outputdir",
        type=Path,
        help="Path to location to save output dir."
    )

    prompt_details = parser.add_argument_group("Prompt Details")

    prompt_details.add_argument(
        '--prompt-job-description',
        type=Path,
        help='Job Description File for prompt.'
    )
    prompt_details.add_argument(
        '--prompt-template',
        type=Path,
        help='Template file for input prompt.'
    )

    # Provider configuration
    provider_group = parser.add_argument_group('LLM Provider Options')
    provider_group.add_argument(
        "--provider",
        choices=["anthropic", "openai", "together", "deepseek"],
        default="anthropic",
        help="LLM provider to use"
    )
    provider_group.add_argument(
        "--model",
        help="Model name for the selected provider"
    )
    provider_group.add_argument(
        "--api-key",
        type=Path,
        required=True,
        help="Path to api-key_yaml file."
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.api_key.is_file():
        parser.error(f"API Key yaml file not found: {args.api_key}")

    for resume_path in args.resumes:
        if not resume_path.is_file():
            parser.error(f"Resume file not found: {resume_path}")

    if not args.prompt_template.is_file():
        parser.error(f"Prompt Template not found: {args.prompt_template}")

    if not str(args.prompt_template).endswith(('.json', '.txt')):
        parser.error(f"Prompt Template: {args.prompt_template} does not end in json/txt.")


    #args: resumes provider and model
    args.outputdir.mkdir(parents=True, exist_ok=True)
    
    return args

if __name__ == "__main__":
    args = parse_args()

    with open(str(args.api_key), 'r') as file:
        config = yaml.safe_load(file)
    
    user_input_api_key = config['services'][args.provider]['api_key']
 
    #Format parse-args to input into client initialization below. 
    prompt_template_path = args.prompt_template
    formatted_prompt_template = read_template(prompt_template_path)

    prompt_job_description_name = args.prompt_job_description.name if args.prompt_job_description else None
    prompt_job_description_str = open(args.prompt_job_description, 'r').read() if args.prompt_job_description else None

    #Define Client
    if args.provider == 'together':
        client = TogetherAIClient(input_api_key = user_input_api_key,
                                  model=args.model, 
                                  prompt_template=formatted_prompt_template,
                                  prompt_template_path=str(prompt_template_path),
                                  prompt_job_desc_filename=prompt_job_description_name,
                                  prompt_job_description = prompt_job_description_str)
    elif args.provider == 'anthropic':
        client = AnthropicClient(input_api_key = user_input_api_key,
                                  model=args.model, 
                                  prompt_template=formatted_prompt_template,
                                  prompt_template_path=str(prompt_template_path),
                                  prompt_job_desc_filename=prompt_job_description_name,
                                  prompt_job_description = prompt_job_description_str)
    elif args.provider == 'openai':
        client = OpenAIClient(input_api_key = user_input_api_key,
                                  model=args.model, 
                                  prompt_template=formatted_prompt_template,
                                  prompt_template_path=str(prompt_template_path),
                                  prompt_job_desc_filename=prompt_job_description_name,
                                  prompt_job_description = prompt_job_description_str)
    elif args.provider == 'deepseek':
        client = DeepSeekClient(input_api_key=user_input_api_key,
                              model=args.model,
                              prompt_template=formatted_prompt_template,
                              prompt_template_path=str(prompt_template_path),
                              prompt_job_desc_filename=prompt_job_description_name,
                              prompt_job_description=prompt_job_description_str)
    else:
        raise ValueError("Provider client not found")

    if str(prompt_template_path).endswith('.txt'):
        prompt_template_path_name = str(prompt_template_path)[:-4]  # Remove .txt
    else:
        prompt_template_path_name = str(prompt_template_path)[:-5]
    
    for resume_path in args.resumes:

        #Make neccesary folders to store output csvs.
        saved_dir_ppaths = [
            f"{args.outputdir}/input_cvs_{str(resume_path)[:-4]}",
            f"{args.outputdir}/input_cvs_{str(resume_path)[:-4]}/prompt_template_{prompt_template_path_name}",
         ]
        if args.prompt_job_description is not None:
            saved_dir_ppaths.append(f"{args.outputdir}/input_cvs_{str(resume_path)[:-4]}/{args.provider}_{client.model}/prompt_template_{prompt_template_path_name}/job_desc_{args.prompt_job_description.name[:-4]}")
       
       
        for path in saved_dir_ppaths:
            if not os.path.isdir(path):
                Path(path).mkdir(parents=True, exist_ok=True)

        #Generate and save modified resumes.
        modified_resumes = client.generate_group_of_cv_s(cv_s_dataframe=pd.read_csv(str(resume_path), index_col=0))
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d_%H-%M')

        new_file_name = f"file_{timestamp_str}.csv"
        if modified_resumes is not None:
            modified_resumes.to_csv(f"{saved_dir_ppaths[-1]}/{new_file_name}")
        else:
            print(f"No modified resumes outputted for this {resume_path}. Refer to previous error logs.")

