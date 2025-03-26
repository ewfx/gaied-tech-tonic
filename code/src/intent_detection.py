import json
import os
from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from pprint import pprint

current_dir = os.path.dirname(__file__)
intents_mappings = os.path.join(current_dir, "../data/mappings.json")
expected_output = os.path.join(current_dir, "../data/expected_output.json")
fields_to_be_extracted = os.path.join(current_dir, "../data/fields.json")

# Load Mistral Model from Hugging Face
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

agent = Agent(
    model=HuggingFace(
        id=model_name,
        max_tokens=4096,
        response_format={ "type": "json_object" }
    ),
    markdown=True
)

# Load intent mappings from the JSON file
with open(intents_mappings, "r") as f:
    intent_to_business_mapping = json.load(f)

    
# Load intent mappings from the JSON file
with open(fields_to_be_extracted, "r") as f:
    fields_list = json.load(f)

def detect_intent(email_text):
    print("\nDetecting intent...")
    #Classify the email into a predefined banking intent.
    prompt = f"""
    You are a banking assistant. Your task is to classify the following email into one of the following predefined intent categories. 
    Each intent category will have ask field which should corelate with the email content. If you find any relavent info as per the ask, then extract the coresponding request type and sub request type from the same Intent Category list below.
    
    Email Content:
    "{email_text}"

    Intent Categories:
    {json.dumps(intent_to_business_mapping, indent=2)}
    
    Please give **only** one set of request type and sub request type in the below json format if found. Don't deviate from the structure of the json format and don't add any illusanation statement.
    Example JSON format: {{"request_type": "Test Request Type", "sub_request_type": "Test Sub Request Type"}}

    """
    intent_result = agent.run(prompt)
    pprint(f"intent_result: {intent_result.content}")

    prompt = f"""
    Focus on extracting the fields using {json.dumps(fields_list, indent=2)}, even if they are expressed in different words or formats:
    Each field includes its name, description, acronyms, and special instructions to help you identify it in the email content.

    Please note that fields mentioned above are primary fields to consider. So, if you find some other fields required for business needs, extract them too. Extract the fields in json format.

    Email Content:
    "{email_text}"

    Provide **only** the list of extracted fields in the exact JSON format below. Do not include any additional text, explanations, or statements. Only return the JSON object. What i mean is that don't include statements such as Here's the extracted fields or anything similar.
    Example JSON format: {{"Deal Name": "XYZ Corporation", "Requested Amount": "$100,000"}}

    """
    extracted_fields = agent.run(prompt)
    pprint(f"extracted_fields: {extracted_fields.content}")
    return {
        "intent": intent_result.content,
        "extracted_fields": extracted_fields.content,
    }
