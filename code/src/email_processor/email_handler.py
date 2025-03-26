from mailparser import parse_from_file

from email_processor.attachment_handler import process_attachments

def extract_email(eml_path: str) -> str:
    email = parse_from_file(eml_path)
    email_content = email.body  # Extract email body

    # Process attachments
    attachments_content = process_attachments(email)

    # Combine email body and attachment content
    full_content = email_content + "\n" + attachments_content
    return full_content