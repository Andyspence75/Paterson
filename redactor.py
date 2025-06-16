
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def redact_text(text):
    doc = nlp(text)
    redacted = []
    for token in doc:
        if token.ent_type_ in ["PERSON", "GPE", "ORG", "LOC"]:
            redacted.append("[REDACTED]")
        else:
            redacted.append(token.text)
    return " ".join(redacted)
