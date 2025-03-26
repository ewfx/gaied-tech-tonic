from transformers import AutoTokenizer, pipeline

# Load Mistral Model from Hugging Face
####not working on 8GB RAM####
# model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
# model_name = "deepset/roberta-base-squad2"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
####not working on 8GB RAM####
# model_name = "google/gemma-1.1-2b-it" #slow and not accurate
# model_name = "facebook/bart-large-mnli" 
model_name = "distilbert/distilbert-base-uncased"

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
