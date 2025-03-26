from preprocess import preprocess_email, extract_named_entities
from intent_detection2 import detect_intent
from data_extraction import extract_structured_data

print("\n🚀 Script Started...\n", flush=True)

def process_email(email_text):
    print("\n🔹 Processing Email...")
    """Run the full pipeline: NLP -> Intent Detection -> Data Extraction."""
    # Preprocessing
    cleaned_text = preprocess_email(email_text)
    entities = extract_named_entities(email_text)
    print("\n🔹 Extracted Entities:\n", entities)

    # Intent Detection
    detected_intent = detect_intent(cleaned_text)

    # Structured Data Extraction
    extracted_data = extract_structured_data(email_text, detected_intent)

    return {
        "entities": entities,
        "intent": detected_intent,
        "extracted_data": extracted_data,
    }

if __name__ == "__main__":
    print("\n🔹 Email Processing Pipeline")
    sample_email = """
    BANK OF XYZ  
Bank of XYZ, N.A.  
To:                                                                                ABC BANK NATIONAL ASSOCIATION  
Date:                                                                              8 - Nov - 2023  
ATTN:                                                                              AGENT DEFAULT  
Phone:                                                                             123-456-7890  
Fax:                                                                               987-654-3210  
Email:                                                                             QWE123@randommail.com  
Re:                                                                                RANDOM ENTITY LP USD 999MM MAR22 / REVOLVER / ENTITY XYZ00099  
Deal CUSIP:                                                                        A1B2C3D4E5F6  
Deal ISIN:                                                                         USA1B2C3D4E5  
Facility CUSIP:                                                                    G7H8I9J0K1L2  
Facility ISIN:                                                                     USG7H8I9J0K1  
Lender MEI:                                                                        USX1Y2Z3A4B5  
Effective 10 - Nov - 2023, RANDOM ENTITY LP has elected to repay under the SOFR (US) Term option, a total of USD  
99,999,999.99.  
Previous Global Principal Balance: USD 199,999,999.99  
New Global Principal Balance: USD 99,999,999.99  
This loan was effective 20 - Jul - 2023 and is scheduled to reprice on 20 - Nov - 2023.  
Your share of the USD 99,999,999.99 SOFR (US) Term option payment is USD 9,876,543.21  
Previous Lender Share Principal Balance: USD 19,876,543.21  
New Lender Share Principal Balance: USD 9,876,543.21  
We will remit USD 9,876,543.21 on the effective date. Please note that (1) If the Borrower has not in fact made such  
payment; or (ii) any payment you receive is in excess of what was paid by the Borrower or (iii) we notify you that the payment  
was erroneously made, then pursuant to the provisions of the credit facility, you agree to return such payment.  

For: ABC BANK  
To: ABC BANK, N.A.  
ABA Number: 111111  
Account No: XXXXXXXXXX5678  
Reference: RANDOM ENTITY LP USD 999MM MAR22, SOFR (US) Term Principal Payment (ENTITY XYZ00099)  

Thanks & Regards,  
John Doe  
Telephone: +19999999999  
Email ID: johndoe@randommail.com  
    """
    result = process_email(sample_email)
    print(result)
