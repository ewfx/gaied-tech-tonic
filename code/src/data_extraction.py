from agno.agent import Agent
from agno.models.huggingface import HuggingFace

# Initialize Agno Agent with Mistral
agent = Agent(
    model=HuggingFace(
        id="mistralai/Mistral-7B-Instruct-v0.1",
        max_tokens=4096,
    ),
    markdown=True
)

def extract_structured_data(email_text, detected_intent):
    """Extract key details like Deal Name, Requested Amount, Effective Date, etc."""
    prompt_details = f"""
    You are an AI assistant tasked with extracting structured data from emails related to commercial banking and lending. 
    Focus on extracting the following fields, even if they are expressed in different words or formats:

    - **Deal Name**: The name of the deal, transaction, or agreement. Mostly this will be in Capital letters.
    - **Requested Amount**: The amount requested in the transaction (e.g., loan amount).
    - **Requested Currency**: The currency in which the requested amount is denominated.
    - **Lender Currency**: The currency in which the lender is providing the funds.
    - **Lender Amount**: The amount provided by the lender.
    - **Effective Date**: The date when the transaction or agreement becomes effective.
    - **Agent Bank**: The name of the bank acting as the agent or managing bank.
    - If any other additional fields found, then extract them too with likely name with its value.

    Please extract these fields from the following email:
    ---
    {email_text}
    ---
    Detected Intent: {detected_intent}

    Ensure the output is in JSON format with the following structure:
    {{
        "Deal Name": "<value>",
        "Requested Amount": "<value>",
        "Requested Currency": "<value>",
        "Lender Currency": "<value>",
        "Lender Amount": "<value>",
        "Effective Date": "<value>",
        "Agent Bank": "<value>",
        "CUSIP": "<value>",
        "ISIN": "<value>",
        "Other Relevant Details": "<value>"
    }}
    If a field is not found, set its value to `null`.
    """
    return agent.run(prompt_details)