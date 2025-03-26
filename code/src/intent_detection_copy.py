from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
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
    You are a banking assistant. Your task is to classify the following email into one of these predefined intent categories using "ask" in {json.dumps(intent_to_business_mapping, indent=2)}. Using ask, extract requestType and subRequestType fields.
    
    Email Content:
    "{email_text}"
    
    """
    intent_result = agent.run(prompt)
    pprint(f"intent_result: {intent_result}")

    prompt = f"""
    Focus on extracting the fields using {json.dumps(fields_list, indent=2)}, even if they are expressed in different words or formats:
    Each field includes its name, description, acronyms, and special instructions to help you identify it in the email content.

    Please note that below fields are primary fields to consider. So, if you find some other fields required for business needs, extract them too. Extract the fields in json format.

    Email Content:
    "{email_text}"
    """
    extracted_fields = agent.run(prompt)
    pprint(f"extracted_fields: {extracted_fields}")
    return intent_result
    # detected_intent = intent_result.content.strip().strip('"')

    # print(f"Detected Intent: {detected_intent}")

    # # Map the detected intent to a business request type
    # business_request = intent_to_business_mapping.get(detected_intent, "Unknown Request Type")

    # # Extract Request Type and Sub Request Type
    # if ">" in business_request:
    #     request_type, sub_request_type = map(str.strip, business_request.split(">", 1))
    # else:
    #     request_type, sub_request_type = business_request, None

    # print(f"Request Type: {request_type}")
    # print(f"Sub Request Type: {sub_request_type}")

    # return {
    #     "business_request": detected_intent,
    #     "request_type": request_type,
    #     "sub_request_type": sub_request_type,
    # }