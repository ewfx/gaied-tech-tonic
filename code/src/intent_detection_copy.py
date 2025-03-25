from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import os
from agno.agent import Agent
from agno.models.huggingface import HuggingFace

current_dir = os.path.dirname(__file__)
intents_path = os.path.join(current_dir, "../data/intents.json")
intents_mappings = os.path.join(current_dir, "../data/intent_mappings.json")
fields_to_be_extracted = os.path.join(current_dir, "../data/fields.json")

print("\n🔹 intents_path:", intents_path)

# Load Mistral Model from Hugging Face
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "facebook/bart-large-mnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# intent_classifier = pipeline("zero-shot-classification", model=model_name)

agent = Agent(
    model=HuggingFace(
        id=model_name,
        max_tokens=4096,
        response_format={ "type": "json_object" }
    ),
    markdown=True
)

# Define candidate intents
with open(intents_path, "r") as f:
    intents_data = json.load(f)
intents = intents_data["intents"]
print("\n🔹 Intents:", intents)

# Load intent mappings from the JSON file
with open(intents_mappings, "r") as f:
    intent_to_business_mapping = json.load(f)

    
# Load intent mappings from the JSON file
with open(fields_to_be_extracted, "r") as f:
    fields_list = json.load(f)

def detect_intent(email_text):
    print("\nDetecting intent...")
    """Classify the email into a predefined banking intent."""
     # Construct the refined prompt
    prompt = f"""
    You are a banking assistant. Your task is to classify the following email into one of these predefined intent categories:

    {json.dumps(intents, indent=2)}

    Email Content:
    "{email_text}"

    Focus on extracting the following fields, even if they are expressed in different words or formats:
    Below are the master fields list that you need to extract from the email. Each field includes its name, description, acronyms, and special instructions to help you identify it in the email content:
    Please note that below fields are primary fields to consider. So, if you find some other critical fields found that are critical for business needs, please extract them too.
    Master Fields List:
    {json.dumps(fields_list, indent=2)}
    

    Respond **only** with the intent label and list of identified fields with its values..
    """
    intent_result = agent.run(prompt)
    detected_intent = intent_result.content.strip().strip('"')

    print(f"Detected Intent: {detected_intent}")

    # Map the detected intent to a business request type
    business_request = intent_to_business_mapping.get(detected_intent, "Unknown Request Type")

    # Extract Request Type and Sub Request Type
    if ">" in business_request:
        request_type, sub_request_type = map(str.strip, business_request.split(">", 1))
    else:
        request_type, sub_request_type = business_request, None

    print(f"Request Type: {request_type}")
    print(f"Sub Request Type: {sub_request_type}")

    return {
        "business_request": detected_intent,
        "request_type": request_type,
        "sub_request_type": sub_request_type,
    }