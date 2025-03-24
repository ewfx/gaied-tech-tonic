import spacy
import re
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import pipeline
from agno.agent import Agent, RunResponse
from agno.models.huggingface import HuggingFace

# ==============================
# ✅ STEP 1: Initialize Agno with Mistral LLM
# ==============================
agent = Agent(
    model=HuggingFace(
        id="mistralai/Mistral-7B-Instruct-v0.1",
        max_tokens=1024,  # Adjust based on requirement
        # api_key="",  # Replace with your HF API Key if needed
    ),
    markdown=True
)

# ==============================
# ✅ STEP 2: Load NLP Models (SpaCy & Flair)
# ==============================
# Load SpaCy for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Load Flair for Enhanced NER
flair_tagger = SequenceTagger.load("flair/ner-english")

# Load Hugging Face model for Zero-Shot Intent Classification
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ==============================
# ✅ STEP 3: Sample Email Input
# ==============================
email_text = """
Dear Support,
I want to pay USD 1000 on 04-01-25.
Let me know if anything else is needed.
Thanks,
John Doe
"""

# email_text = input("Please enter the email content:\n")

# ==============================
# ✅ STEP 4: Structured Data Extraction (SpaCy + Flair)
# ==============================
def extract_entities(text):
    """ Extract Amount & Effective Date using SpaCy & Regex """
    doc = nlp(text)
    extracted_data = {"Amount": None, "Effective Date": None, "Entities": []}

    # Extract monetary values
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            extracted_data["Amount"] = ent.text  # Extract amount

    # Extract date using regex (as SpaCy may not detect all formats)
    date_match = re.search(r"\d{2}-\d{2}-\d{2}", text)
    if date_match:
        extracted_data["Effective Date"] = date_match.group()

    return extracted_data

# Extract structured details
structured_data = extract_entities(email_text)

# ==============================
# ✅ STEP 5: Entity Recognition with Flair
# ==============================
def extract_flair_entities(text):
    """ Extract entities using Flair for better accuracy """
    sentence = Sentence(text)
    flair_tagger.predict(sentence)

    entities = []
    for entity in sentence.get_spans("ner"):
        entities.append((entity.text, entity.tag))
    
    return entities

# Extract Flair entities
flair_entities = extract_flair_entities(email_text)
structured_data["Entities"] = flair_entities

# ==============================
# ✅ STEP 6: Intent Detection (Mistral via Hugging Face)
# ==============================
# Define candidate intents
intents = ["Money movement inbound", "Money movement outbound", "Account closure", "General inquiry"]

# Perform intent detection
intent_result = intent_classifier(email_text, candidate_labels=intents)
detected_intent = intent_result["labels"][0]  # Get the highest-scoring intent

# ==============================
# ✅ STEP 7: Use Mistral via Agno for Contextual Extraction
# ==============================
prompt_details = f"""
Extract key details (amount and effective date) from the following email:
"{email_text}"

Detected Intent: {detected_intent}

Ensure the response is in structured JSON format.

Response:
"""
extracted_data = agent.run(prompt_details)

# ==============================
# ✅ STEP 8: Final Output
# ==============================
print("\n🔹 **Detected Intent:**", detected_intent)
print("\n🔹 **Structured Data (SpaCy + Flair):**", structured_data)
print("\n🔹 **Extracted Data (Mistral via Agno):**", extracted_data)
