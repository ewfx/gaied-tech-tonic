from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

os.environ["HF_TOKEN"] = ""

# Load Mistral Model from Hugging Face
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
intent_classifier = pipeline("zero-shot-classification", model=model_name)

# Define candidate intents
intents = [
    "Money movement inbound",
    "Money movement outbound",
    "Account closure",
    "General inquiry",
    "Loan repayment",
    "Balance transfer",
]

def detect_intent(email_text):
    print("\nDetecting intent...")
    """Classify the email into a predefined banking intent."""
    intent_result = intent_classifier(email_text, intents)
    return intent_result["labels"][0]  # Highest scoring intent
