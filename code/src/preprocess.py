import spacy
import re

# Load SpaCy NLP Model
nlp = spacy.load("en_core_web_sm")

# Configurable fields to extract with patterns
FIELD_CONFIG = {
    "Deal Name": {
        "entity_labels": ["ORG", "PRODUCT"],
        "keywords": ["deal name", "transaction name", "agreement"],
    },
    "Requested Amount": {
        "entity_labels": ["MONEY"],
        "keywords": ["requested amount", "amount requested", "loan amount"],
    },
    "Requested Currency": {
        "entity_labels": ["CURRENCY"],
        "keywords": ["requested currency", "currency requested"],
    },
    "Lender Currency": {
        "entity_labels": ["CURRENCY"],
        "keywords": ["lender currency", "currency lender"],
    },
    "Lender Amount": {
        "entity_labels": ["MONEY"],
        "keywords": ["lender amount", "amount lender"],
    },
    "Effective Date": {
        "entity_labels": ["DATE"],
        "keywords": ["effective date", "start date", "commencement date"],
    },
    "Agent Bank": {
        "entity_labels": ["ORG"],
        "keywords": ["agent bank", "bank agent", "managing bank"],
    },
}

def preprocess_email(text):
    """Tokenize, remove stopwords, and clean email content."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_named_entities(text, field_config=FIELD_CONFIG):
    """
    Extract important entities based on the configured fields.
    :param text: The email content to process.
    :param field_config: A dictionary mapping field names to SpaCy entity labels and keywords.
    :return: A dictionary of extracted fields and their values.
    """
    doc = nlp(text)
    extracted_fields = {}

    # Iterate over the configured fields
    for field, config in field_config.items():
        # Extract entities matching the configured labels
        entity_matches = [
            ent.text for ent in doc.ents if ent.label_ in config["entity_labels"]
        ]

        # Extract values based on keywords
        keyword_matches = []
        for keyword in config["keywords"]:
            # Use regex to find keyword matches in the text
            match = re.search(rf"\b{keyword}\b.*?:?\s*(\S.*?)(?:\.|,|$)", text, re.IGNORECASE)
            if match:
                keyword_matches.append(match.group(1).strip())

        # Combine entity matches and keyword matches, prioritizing keyword matches
        extracted_values = keyword_matches or entity_matches
        extracted_fields[field] = extracted_values[0] if extracted_values else None

    return extracted_fields