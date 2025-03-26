from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Define candidate intents
intents = [
    "Advance + Current is empty",
    "Advance + Currency = USD",
    "Advance + Currency = Non USD",
    "Payment > Principal",
    "Payment > Interest",
    "Payment > Fee",
    "Payment > Principal & Interest",
    "Reprice",
    "Letter of Credit > Decrease",
    "Letter of Credit > Increase",
    "Letter of Credit > Issuance",
    "Letter of Credit > Extension",
    "Letter of Credit > Termination",
    "Letter of Credit > Fee Payment",
    "Adjustment",
    "Commitment Change",
]

# Mapping of intents to business request types and subtypes
intent_to_business_mapping = {
    "Advance + Current is empty": "Money Movement - Outbound",
    "Advance + Currency = USD": "Money Movement - Outbound > Standard",
    "Advance + Currency = Non USD": "Money Movement - Outbound > Foreign Currency/FX",
    "Payment > Principal": "Money Movement - Inbound > Principal",
    "Payment > Interest": "Money Movement - Inbound > Interest",
    "Payment > Fee": "Fee Payment",
    "Payment > Principal & Interest": "Money Movement - Inbound > Principal + Interest",
    "Reprice": "Rate > Rate Set",
    "Letter of Credit > Decrease": "Letter of Credit > Decrease",
    "Letter of Credit > Increase": "Letter of Credit > Increase",
    "Letter of Credit > Issuance": "Letter of Credit > Issuance",
    "Letter of Credit > Extension": "Letter of Credit > Extension",
    "Letter of Credit > Termination": "Letter of Credit > Termination",
    "Letter of Credit > Fee Payment": "Fee Payment",
    "Adjustment": "Rate > Rate Adjustment/Correction",
    "Commitment Change": "Commitment Changes",
}

# Load Mistral Model from Hugging Face
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
intent_classifier = pipeline("zero-shot-classification", model=model_name)

# Define candidate intents
# intents = [
#     "Money movement inbound",
#     "Money movement outbound",
#     "Account closure",
#     "General inquiry",
#     "Loan repayment",
#     "Balance transfer",
# ]

def detect_intent(email_text):
    print("\nDetecting intent...")
    """Classify the email into a predefined banking intent."""
    intent_result = intent_classifier(email_text, intents)
    detected_intent = intent_result["labels"][0]  # Highest scoring intent
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