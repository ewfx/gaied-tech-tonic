from transformers import AutoTokenizer, pipeline
from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from instructions import revised_instructions
import json
import os

# Load Mistral Model from Hugging Face
####not working on 8GB RAM####
model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
# model_name = "deepset/roberta-base-squad2"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
####not working on 8GB RAM####
# model_name = "google/gemma-1.1-2b-it" #slow and not accurate
# model_name = "facebook/bart-large-mnli" 
# model_name = "distilbert/distilbert-base-uncased"

current_dir = os.path.dirname(__file__)
intents_path = os.path.join(current_dir, "../data/intents.json")


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

def detect_intent(email_text):
    print("\nDetecting intent...")

    # Construct the refined prompt
    prompt = f"""
    You are a banking assistant. Your task is to classify the following email into one of these predefined intent categories:

    {json.dumps(intents, indent=2)}

    Email:
    "{email_text}"

    Respond **only** with the intent label from the list above.
    """

    # Run the model
    intent_result = agent.run(prompt)

    top_intent = intent_result.content.strip().strip('"')
    print("\n🔹 top_intent:",top_intent)
    return top_intent





