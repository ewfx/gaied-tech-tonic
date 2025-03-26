import json
from preprocess import preprocess_email, extract_named_entities
from intent_detection_copy import detect_intent
from data_extraction import extract_structured_data
from agno.utils.pprint import pprint_run_response
from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from pprint import pprint
from pathlib import Path
from email_processor.email_handler import extract_email

print("\n🚀 Script Started...\n", flush=True)

agent = Agent(
    model=HuggingFace(
        id="mistralai/Mistral-7B-Instruct-v0.1",
        max_tokens=4096,
    ),
    markdown=True
)

def process_email(email_text):
    print("\n🔹 Processing Email...")
    """Run the full pipeline: NLP -> Intent Detection -> Data Extraction."""
    # Preprocessing
    cleaned_text = preprocess_email(email_text)

    # Intent Detection
    detected_intent_fields = detect_intent(cleaned_text)

    # Structured Data Extraction with cleaned email text
    # print("\n🔹 Fields Detected with cleaned email text", detected_intent_fields)
    # extracted_data_cleaned = extract_structured_data(cleaned_text, detected_intent_fields)
    # extracted_data_cleaned_json = extracted_data_cleaned.content  # Extract JSON content
    # pprint(extracted_data_cleaned_json)

    return {
        "Final Data": detected_intent_fields,
    }

def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / "input"   
    output_path = script_dir / "output"
    if not input_path.exists():
        print(f"Input folder does not exist: {input_path}")
        return
    
    if not output_path.exists():
        output_path.mkdir()  # Create the output folder if it doesn't exist

    print("\n🔹 Email Processing Pipeline")
    # Check if there are any .eml files in the input folder
    email_files = list(input_path.glob("*.eml"))
    if not email_files:
        print(f"No email files found in the provided path: {input_path}")
        return

    # Iterate through all .eml files in the input folder
    for email_file in email_files:
        email_data = extract_email(email_file)
        result = process_email(email_data)
        print("\n🔹 Full Processing Result:")
        pprint(result)
        # Write the result to a file in the output folder
        output_file = output_path / f"{email_file.stem}_result.json"  # Save as JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print(f"Result written to: {output_file}")

if __name__ == "__main__":
    main()