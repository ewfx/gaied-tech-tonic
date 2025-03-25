from preprocess import preprocess_email, extract_named_entities
from intent_detection import detect_intent
from data_extraction import extract_structured_data

print("\n🚀 Script Started...\n", flush=True)

def process_email(email_text):
    print("\n🔹 Processing Email...")
    """Run the full pipeline: NLP -> Intent Detection -> Data Extraction."""
    # Preprocessing
    cleaned_text = preprocess_email(email_text)
    print("\n🔹 Preprocessed Text:\n", cleaned_text)
    entities = extract_named_entities(email_text)
    print("\n🔹 Extracted Entities:\n", entities)

    # Intent Detection
    detected_intent = detect_intent(cleaned_text)
    print("\n🔹 Detected Intent:", detected_intent)

    # Structured Data Extraction
    extracted_data = extract_structured_data(email_text, detected_intent)
    print("\n🔹 Extracted Data:\n", extracted_data)

    return {
        "cleaned_text": cleaned_text,
        "entities": entities,
        "intent": detected_intent,
        "extracted_data": extracted_data,
    }

if __name__ == "__main__":
    print("\n🔹 Email Processing Pipeline")
    sample_email = """
    Effective 10-Nov-2023, CANTOR FITZGERALD LP has elected to repay under the SOFR (US) Term option, a total of USD 20,000,000.00.
    Previous Global principal balance: USD 45,000,000.00
    New Global principal balance: USD 25,000,000.00
    """
    result = process_email(sample_email)
    print("\n🔹 Full Processing Result:\n", result)
