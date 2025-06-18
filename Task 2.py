import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Example Amazon reviews
reviews = [
    "I love my new Samsung Galaxy phone. The battery life is amazing!",
    "The Sony headphones broke after a week. Very disappointed.",
    "Apple MacBook is fast and reliable. Highly recommend!",
    "This HP printer is terrible. It keeps jamming.",
    "The Nike shoes are comfortable and stylish."
]

# Simple positive and negative word lists for sentiment
positive_words = {"love", "amazing", "fast", "reliable", "highly recommend", "comfortable", "stylish"}
negative_words = {"broke", "disappointed", "terrible", "jamming"}

def analyze_sentiment(text):
    text_lower = text.lower()
    # Check for positive/negative keywords
    if any(word in text_lower for word in positive_words):
        return "Positive"
    elif any(word in text_lower for word in negative_words):
        return "Negative"
    else:
        return "Neutral"

for review in reviews:
    doc = nlp(review)
    # Extract entities labeled as ORG (brands) or PRODUCT (product names, if available)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT")]
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}")
    print(f"Extracted Entities (Product/Brand): {entities}")
    print(f"Sentiment: {sentiment}")
    print("-" * 60)
