# 🚀 Project Name
gaied-tech-tonic
## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
**Problem Statement**

Commercial Bank Lending Service teams receive a high volume of servicing requests through emails. These emails contain diverse requests, often with attachments, which must be ingested into the loan servicing platform to create service requests. These requests then undergo workflow processing.

Currently, incoming service requests via email require a manual triage process performed by a "Gatekeeper," who:

Reads and interprets email content and attachments.

Identifies the intent of the email and classifies it into predefined "Request Type" and "Sub Request Type."

Extracts key attributes for service request creation.

Assigns the request to the appropriate team or individual based on roles and skills.

This manual triage process requires significant human effort, is time-consuming, and becomes inefficient and error-prone when dealing with large email volumes.

**Solution Overview**

To address these challenges, this project implements a GenAI-powered Email Classification and OCR Solution. The solution automates the classification and data extraction from interbank emails and attachments using Generative AI (LLMs). This improves efficiency, accuracy, and turnaround time while minimizing manual gatekeeping efforts.


## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
Likes doing hackathon where I can explore new technologies.
Current manual triage process requires significant human effort, is time-consuming, and becomes inefficient and error-prone when dealing with large email volumes.

## ⚙️ What It Does
**Key Features:**

**Automated Email Processing:** Uses NLP and AI models to extract and classify banking transactions.

**High Accuracy:** Leverages advanced LLMs like Mistral for precise intent detection and structured data extraction.

**Scalability & Performance:** Supports large volumes of emails while ensuring fast and efficient processing.

**Regulatory Compliance:** Helps banks maintain compliance by accurately extracting and verifying financial transaction details.

**Multi-Request Handling:** Identifies multiple request types in a single email and prioritizes the primary request.

**Duplicate Detection:** Prevents redundant service requests by flagging duplicate emails.

## 🛠️ How We Built It
**Solution Architecture**

**Email Preprocessing & Named Entity Recognition (NER)** → Cleans email text and extracts key entities (e.g., amount, bank details, transaction IDs).

**Intent Detection using LLM** → Identifies request types (e.g., loan repayments, settlements, fund transfers).

**Structured Data Extraction** → Maps extracted financial details to standard banking services.

**Duplicate Detection** → Flags duplicate emails to avoid redundant processing.

**Fine-Tuning & Continuous Learning** → Improves accuracy through historical data training.

**Architecture Diagram**

┌───────────────────────┐     ┌───────────────────────┐     ┌──────────────────────────┐
│ Email Preprocessing  │ →  │ Intent Detection      │ →  │ Structured Data Extraction │
└───────────────────────┘     └───────────────────────┘     └──────────────────────────┘
                           ↓
              ┌───────────────────────────┐
              │ Duplicate Email Detection │
              └───────────────────────────┘
                           ↓
              ┌───────────────────────────┐
              │ Fine-Tuning with Historical Data │
              └───────────────────────────┘


## 🚧 Challenges We Faced
LLM model Mistral-7B needs needs high configuration, we had to explore and spend lot of time on optimizing the solution.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/your-repo.git
   ```
2. Install dependencies  
   a. Create virtual environment (say 'venv')
   b. venv\Scripts\activate
   
   ```sh
      pip install -r requirements.txt (for Python)
      python -m spacy download en_core_web_sm
   ```
4. Run the project  
   ```sh
   python .\main.py
   ```

## 🏗️ Tech Stack
- 🔹 Backend: Python / FastAPI / Mistral(LLM)

## 👥 Team
- **Your Name** - [navakanth09-OL](https://github.com/navakanth09-OL) | navakanth09
- **Pavan Kumar Neeli** - [https://github.com/pavannpg | [[LinkedIn](#)](https://www.linkedin.com/in/pavankumarneeli/)
