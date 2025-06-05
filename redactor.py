
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def redact_text(text):
    doc = nlp(text)
    redacted = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "LOC"]:
            redacted = redacted.replace(ent.text, "[REDACTED]")
    return redacted
