import os
import base64
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from pytesseract import image_to_string

def extract_attachment_content(file_path: str) -> str:
    content = ""
    print("Extracting content from attachment:", file_path)
    file_extension = file_path.split(".")[-1].lower()

    if file_extension == "pdf":
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text()
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")

    elif file_extension in ["doc", "docx"]:
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading Word file {file_path}: {e}")

    elif file_extension in ["png", "jpg", "jpeg"]:
        try:
            image = Image.open(file_path)
            content += image_to_string(image)
        except Exception as e:
            print(f"Error reading image file {file_path}: {e}")

    else:
        print(f"Unsupported attachment type: {file_extension}")

    return content


def process_attachments(email) -> str:
    # Clear the output folder before processing new attachments
    script_dir = Path(__file__).parent  # Get the directory of the current script
    output_folder = script_dir.parent / "output"
   
    attachments_content = ""
    for attachment in email.attachments:
        attachment_name = attachment["filename"]
        attachment_data = attachment["payload"]

        # Save attachment to a temporary file
        temp_file_path = os.path.join(output_folder, attachment_name)
        try:
            with open(temp_file_path, "wb") as temp_file:
                # Decode the payload if it is Base64-encoded
                if attachment.get("content_transfer_encoding") == "base64":
                    temp_file.write(base64.b64decode(attachment_data))
                else:
                    # Write the raw binary data directly
                    if isinstance(attachment_data, str):
                        temp_file.write(attachment_data.encode("utf-8"))
                    else:
                        temp_file.write(attachment_data)

            # Confirm the file was saved
            if os.path.exists(temp_file_path):
                print(f"Attachment saved: {temp_file_path}")
            else:
                print(f"Failed to save attachment: {temp_file_path}")
                continue  # Skip processing this attachment

            # Extract content from the attachment
            attachments_content += extract_attachment_content(temp_file_path) + "\n"

        except Exception as e:
            print(f"Error saving attachment {attachment_name}: {e}")
            continue  # Skip this attachment

        finally:
            # Remove the temporary file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return attachments_content