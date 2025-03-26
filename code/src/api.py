from fastapi import FastAPI
from pydantic import BaseModel
from main import process_email

app = FastAPI()

class EmailRequest(BaseModel):
    email_text: str

@app.post("/analyze")
def analyze_email(request: EmailRequest):
    result = process_email(request.email_text)
    return result

# Run server: uvicorn src.api:app --reload
