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
    """Extract amount, effective date, CUSIP, ISIN, etc."""
    prompt_details = f"""
    Extract key details (amount, effective date, CUSIP, ISIN) from the following email:
    {email_text}
    Detected Intent: {detected_intent}
    """
    return agent.run(prompt_details)
