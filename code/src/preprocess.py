import spacy

# Load SpaCy NLP Model
nlp = spacy.load("en_core_web_sm")

def preprocess_email(text):
    """Tokenize, remove stopwords, and clean email content."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_named_entities(text):
    """Extract important entities like names, organizations, dates, and money values."""
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities
