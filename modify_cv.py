"""
This module provides functions to improve resumes/CVs using various LLM APIs.
"""
import time
import argparse
import yaml
import pandas as pd
from abc import ABC, abstractmethod
import format_messages 
from typing import List
from pathlib import Path
from enum import Enum
import logging
import json
from datetime import datetime
from together import Together
import os
from openai import OpenAI
from anthropic import Anthropic

from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

time_sleep_anthropic = 20
time_sleep_openai=20
MAX_TOKENS=1024
openai_individual_batch_threshold=5

class AnthropicClient:
    """Anthropic-specific implementation"""
    def __init__(self, input_api_key:str, model: str, prompt_job_description: str, prompt_type: str, custom_prompt: str):
        self.client = Anthropic(api_key=input_api_key)
        self.model = model if model else "claude-3-sonnet-20240229"
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''
        self.prompt_type = prompt_type if prompt_type else "General Conversation w/o Job Description"
        self.custom_prompt = custom_prompt if custom_prompt else ''
        return 

    def _client_api_call_function_request_input_cv(self, input_cv:str, id: int)->Request:
        messages = format_messages.generate_messages_per_resume(prompt_type=self.prompt_type, input_cv = input_cv, job_description=self.prompt_job_description, custom_prompt=self.custom_prompt)
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
        if len(cv_s_dataframe.columns)>1:
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = "Modified_" + self.model + " "+self.prompt_type+ "_of_" + to_be_modified_col

        original_cv_s = list(cv_s_dataframe[to_be_modified_col])
        cv_batch_requests = [self._client_api_call_function_request_input_cv(input_cv=original_cv_s[i], id=i) for i in range(len(original_cv_s))]

        cv_batch_requests_output = self.client.beta.messages.batches.create(requests=cv_batch_requests)
        batch_id = cv_batch_requests_output.id

        while True:
            message_batch =self.client.beta.messages.batches.retrieve(batch_id)
            if message_batch.processing_status =='ended':
                break
            print(f"Batch {message_batch.id} is still processing...")
            time.sleep(time_sleep_anthropic)

        output_resumes = []
        #Messages Batch Processing is done, ended.
        for result in self.client.beta.messages.batches.results(message_batch.id):
            if result.result.type == 'succeeded':
                output_resumes.append(result.result.message.content[0].text)
            else:
                print(f"Batch of id({result.result.id}) has:"+str(result.result.type))
                output_resumes.append("not_succeeded")
        cv_s_dataframe[modified_col_name] = output_resumes
        return cv_s_dataframe[[modified_col_name]]

class OpenAIClient:
    """OpenAI-specific implementation"""
    def __init__(self, input_api_key: str, model: str, prompt_job_description: str, prompt_type: str, custom_prompt: str):
        self.client = OpenAI(api_key=input_api_key)
        self.model = model if model else 'gpt-4'
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''
        self.prompt_type = prompt_type if prompt_type else 'General Conversation w/o Job Description'
        self.custom_prompt = custom_prompt if custom_prompt else ''
        self.open_ai_batch_id = None

        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime('%Y_%m_%d_%H_%M_%S')
        self.output_file_name = prompt_type+"-"+current_datetime_str+".jsonl"

        self.time_marker = current_datetime_str
        self.num_generated = 0
        return 

    def _client_api_call_function(self, messages)->str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages = messages
        )
        output = response.choices[0].message.content
        return output
    
    def _create_batch_file_input(self, cv_s_dataframe: pd.DataFrame, to_be_modified_col: str = 'CV'):
        #Currently, OpenAI only supports inputs of json-L files.
        original_cv_s = list(cv_s_dataframe[to_be_modified_col])
        batch_inputs = []

        for index, row in cv_s_dataframe.iterrows():
            batch_inputs.append({"custom_id":"resume-request-"+str(index),
                                 'method':"POST",
                                 'url': "/v1/chat/completions",
                                 'body':{'model':self.model, 'messages':format_messages.generate_messages_per_resume(prompt_type=self.prompt_type, input_cv = row[to_be_modified_col], job_description=self.prompt_job_description, custom_prompt=self.custom_prompt)}})

        formatted_inputs_file_name = "openai_formatted_inputs_"+self.time_marker+".jsonl"
        with open(formatted_inputs_file_name, "w") as f:
            for item in batch_inputs:
                f.write(json.dumps(item) + "\n")
        return formatted_inputs_file_name 
    
    def _send_group_of_cv_s_batch(self, cv_s_dataframe: pd.DataFrame, to_be_modified_col: str = 'CV'):
        batch_input_file_name = self._create_batch_file_input(cv_s_dataframe=cv_s_dataframe, to_be_modified_col=to_be_modified_col)
        
        batch_input_file = self.client.files.create(file=open(batch_input_file_name, "rb"),purpose="batch")
        batch_input_file_id = batch_input_file.id

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": self.model+"_resume_generation_of_prompt_"+self.prompt_type+"_time_"+self.time_marker
            }
        )

        os.remove(batch_input_file_name)

        self.batch_id = batch_object.id
        return batch_object
    
    def _status_ended(self):
        if self.batch_id:
            if self.client.batches.retrieve(self.batch_id).status not in ['in_progress', 'validating', 'cancelling', 'finalizing']:
                print(self.client.batches.retrieve(self.batch_id).status)
                return True
            print(f"Batch {self.batch_id} is still "+ self.client.batches.retrieve(self.batch_id).status+".")
            return False
        else:
            raise Exception("Empty batch id.")
            return

    def _cancel_batch_of_cvs(self):
        self.batches.cancel(self.batch_id)
        self.batch_id = None
        return 
        
    def _generate_group_of_cv_s_batch(self, cv_s_dataframe: pd.DataFrame):
        if len(cv_s_dataframe.columns)>1:
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]

        self._send_group_of_cv_s_batch(cv_s_dataframe=cv_s_dataframe, to_be_modified_col=to_be_modified_col)

        if self.batch_id is None:
            raise Exception("There is no batch id - either this job has been previously cancelled or something is wrong.") 
        
        while True:
            if self._status_ended():
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
            #print(file_response.text)

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

            modified_col_name = "Modified_" + self.model + " "+self.prompt_type+ "_of_" + to_be_modified_col
            cv_s_dataframe[modified_col_name] = output_resumes
            return cv_s_dataframe[[modified_col_name]]

    #The following functions do not use the new Batch API functionality. 
    def _generate_individal_cv(self, input_cv: str) -> str:
        # TogetherAI-specific implementation
        messages = format_messages.generate_messages_per_resume(prompt_type = self.prompt_type, input_cv = input_cv, job_description=self.prompt_job_description, custom_prompt = self.custom_prompt)
        output: str = self._client_api_call_function(messages)
        self.num_generated+=1
        print(f"Generated {self.num_generated} resume.")
        return output

    def _generate_group_of_cv_s_from_individual_calls(self, cv_s_dataframe: pd.DataFrame):
        #Generate modified column name
        if len(cv_s_dataframe.columns)>1:
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]
        modified_col_name = "Modified_" + self.model + " "+self.prompt_type+ "_of_" + to_be_modified_col

        #Generate resumes together.
        generate = lambda cv : self._generate_individal_cv(input_cv = cv)
        cv_s_dataframe[modified_col_name]= cv_s_dataframe[to_be_modified_col].apply(generate)
        return cv_s_dataframe[[modified_col_name]]
    
    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame):
        if len(cv_s_dataframe)<=openai_individual_batch_threshold:
            return self._generate_group_of_cv_s_from_individual_calls(cv_s_dataframe)
        else:
            return self._generate_group_of_cv_s_batch(cv_s_dataframe)


class TogetherAIClient:
    """TogetherAI-specific implementation"""
    def __init__(self, input_api_key:str, model: str, prompt_job_description: str, prompt_type: str, custom_prompt: str):
        self.client = Together(api_key=input_api_key)
        self.model = model if model else "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.prompt_job_description = prompt_job_description if prompt_job_description else ''
        self.prompt_type = prompt_type if prompt_type else "General Conversation w/o Job Description"
        self.custom_prompt = custom_prompt if custom_prompt else ''
        self.num_generated = 0
        return 
    
    def _client_api_call_function(self, messages)->str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages = messages
        )
        output = response.choices[0].message.content
        return output

    def _generate_cv(self, input_cv: str) -> str:
        # TogetherAI-specific implementation
        messages = format_messages.generate_messages_per_resume(prompt_type = self.prompt_type, input_cv = input_cv, job_description=self.prompt_job_description, custom_prompt = self.custom_prompt)
        output: str = self._client_api_call_function(messages)
        self.num_generated+=1
        print(f"Generated {self.num_generated} resume.")
        return output

    def generate_group_of_cv_s(self, cv_s_dataframe: pd.DataFrame):
        # TogetherAI-specific batch implementation
        #Generate modified column name
        if len(cv_s_dataframe.columns)>1:
            print(cv_s_dataframe.columns)
            raise Exception("More than one column of resumes inputted. Please reformt input to only contain one column")
        
        to_be_modified_col = cv_s_dataframe.columns[-1]

        modified_col_name = "Modified_" + self.model + " "+self.prompt_type+ "_of_" + to_be_modified_col

        #Generate resumes together.
        generate = lambda cv : self._generate_cv(input_cv = cv)
        cv_s_dataframe[modified_col_name]= cv_s_dataframe[to_be_modified_col].apply(generate)
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
        '--prompt-type',
        type=str,
        choices=['General Conversation w/o Job Description', 'General Conversation w/ Job Description', 'Custom'],
        default = 'General Conversation w/o Job Description',
        help = 'Prompt Type for Modifying Resumes'
    )
    prompt_details.add_argument(
        '--prompt-job-description',
        type=str,
        help='Job Description for prompt.'
    )
    prompt_details.add_argument(
        '--custom-prompt',
        type=str,
        help='Custom Prompt (if type is selected).'
    )

    # Provider configuration
    provider_group = parser.add_argument_group('LLM Provider Options')
    provider_group.add_argument(
        "--provider",
        choices=["anthropic", "openai", "together"],
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

    if args.prompt_type == 'Custom' and args.custom_prompt is None:
        parser.error("Custom prompt type indiciated, but no custom prompt inputted.")
    
    if args.prompt_type == 'General Conversation w/ Job Description' and args.prompt_job_description is None:
        parser.error("Prompt with job description indicated, but no job description inputted. Use --prompt-job-description to put the text in.")

    #args: resumes provider and model
    args.outputdir.mkdir(parents=True, exist_ok=True)
    
    return args

if __name__ == "__main__":
    args = parse_args()

    with open(str(args.api_key), 'r') as file:
        config = yaml.safe_load(file)

    #(self, input_api_key: str, model: str = "claude-3-sonnet-20240229", prompt_job_description: str = '', prompt_type: str='', custom_prompt: str=''):
    user_input_api_key = config['services'][args.provider]['api_key']

    if args.provider == 'together':
        client = TogetherAIClient(input_api_key = user_input_api_key, model=args.model, prompt_type=args.prompt_type, prompt_job_description=args.prompt_job_description, custom_prompt=args.custom_prompt)
    elif args.provider == 'anthropic':
        client = AnthropicClient(input_api_key = user_input_api_key, model=args.model, prompt_type=args.prompt_type, prompt_job_description=args.prompt_job_description, custom_prompt=args.custom_prompt)
    elif args.provider == 'openai':
        client = OpenAIClient(input_api_key = user_input_api_key, model=args.model, prompt_type=args.prompt_type, prompt_job_description=args.prompt_job_description, custom_prompt=args.custom_prompt)
    else:
        raise ValueError("Provider client not found")

    for resume_path in args.resumes:
        modified_resumes = client.generate_group_of_cv_s(cv_s_dataframe=pd.read_csv(str(resume_path), index_col=0))
        new_file_name = "Modified_Model_Type_"+args.provider+"_"+client.model+"_Prompt_Type_"+client.prompt_type+"_Original_File_"+resume_path.name
        new_file_name = new_file_name.replace("/", "_").replace(" ", "_")
        if modified_resumes is not None:
            modified_resumes.to_csv(str(args.outputdir)+"/"+new_file_name)
        else:
            print("No modified resumes outputted. Refer to previous error logs.")

    #Example input
   # python3 modify_cv.py test_csvs.csv test_folder --prompt-type "General Conversation w/o Job Description" --provider together --api-key llm_api_keys.yaml