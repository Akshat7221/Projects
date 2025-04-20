import sounddevice as sd
import nltk
import spacy
import json
from vosk import Model, KaldiRecognizer
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from textblob import TextBlob
import os
from transformers import pipeline  # Import pipeline for sentiment-analysis

# Create a pipeline for sentiment-analysis using RoBERTa
roberta_pipeline = pipeline('sentiment-analysis', model='roberta-base')

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



# Download required datasets from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')


# Function to classify sentence
def classify_sentence(doc):
    independent_clauses = 0
    dependent_clauses = 0

    for token in doc:
        if token.dep_ == 'cc':  # coordinating conjunction (e.g., "and", "but")
            independent_clauses += 1
    if independent_clauses >= 1:
        independent_clauses += 1
    for token in doc:
        if token.dep_ in ['mark', 'relcl', 'advcl', 'ccomp', 'xcomp']:  # handles more dependent clauses
            dependent_clauses += 1

    if independent_clauses >= 2 and dependent_clauses == 0:
        return "Compound Sentence"
    elif independent_clauses == 1 and dependent_clauses >= 1:
        return "Complex Sentence"
    elif independent_clauses >= 2 and dependent_clauses >= 1:
        return "Compound-Complex Sentence"
    else:
        return "Unknown Sentence"  # Added support for simple sentence classification


# Function to count parts of speech
def count_pos_tags(doc):
    pos_count = {
        'Nouns': 0, 'Verbs': 0, 'Adjectives': 0, 'Adverbs': 0,
        'Pronouns': 0, 'Prepositions': 0, 'Conjunctions': 0, 'Interjections': 0
    }
    for token in doc:
        if token.pos_ == 'NOUN':
            pos_count['Nouns'] += 1
        elif token.pos_ == 'VERB':
            pos_count['Verbs'] += 1
        elif token.pos_ == 'ADJ':
            pos_count['Adjectives'] += 1
        elif token.pos_ == 'ADV':
            pos_count['Adverbs'] += 1
        elif token.pos_ == 'PRON':
            pos_count['Pronouns'] += 1
        elif token.pos_ == 'ADP':
            pos_count['Prepositions'] += 1
        elif token.pos_ == 'CCONJ':
            pos_count['Conjunctions'] += 1
        elif token.pos_ == 'INTJ':
            pos_count['Interjections'] += 1
    return pos_count


# Function for Sentiment Analysis
def analyze_sentiment(text, tb_weight=0.4, roberta_weight=0.6):
    # TextBlob polarity
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity

    # RoBERTa classification
    roberta_result = roberta_pipeline(text)[0]
    roberta_label = roberta_result['label']
    roberta_score = roberta_result['score']

    # Convert RoBERTa label to polarity
    if roberta_label == 'LABEL_0':  # Negative
        roberta_polarity = -1 * roberta_score
    elif roberta_label == 'LABEL_1':  # Neutral
        roberta_polarity = 0
    else:  # Positive (LABEL_2)
        roberta_polarity = roberta_score

    # Combine using weights
    final_polarity = (tb_weight * tb_polarity) + (roberta_weight * roberta_polarity)
    sentiment = "Positive" if final_polarity > 0.1 else "Negative" if final_polarity < -0.1 else "Neutral"

    # Print formula and scores
    print("\n--- Sentiment Analysis Details ---")
    print(f"TextBlob Polarity: {tb_polarity:.4f}")
    print(f"RoBERTa Label: {roberta_label}, Raw Score: {roberta_score:.4f}, Converted Polarity: {roberta_polarity:.4f}")
    print(f"Weight Applied -> TextBlob: {tb_weight}, RoBERTa: {roberta_weight}")
    print(f"Formula: Final Polarity = ({tb_weight} × {tb_polarity:.4f}) + ({roberta_weight} × {roberta_polarity:.4f})")
    print(f"Final Weighted Polarity Score: {final_polarity:.4f}")

    return sentiment, round(final_polarity, 4)


# Speech Recognition using Vosk
def get_speech_input(duration=20):
    model_path = r"C:\Users\AKSHAT JAIN\OneDrive\Desktop\project.py\vosk-model-small-en-us-0.15"

    if not os.path.exists(model_path):
        print(
            "Vosk model not found! Download from https://alphacephei.com/vosk/models and extract to the script folder.")
        return ""

    model = Model(model_path)
    samplerate = 16000  # Sampling rate
    print(f"Recording for {duration} seconds... Speak now!")

    # Record voice
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    # Recognize speech
    rec = KaldiRecognizer(model, samplerate)
    rec.AcceptWaveform(recording.tobytes())
    result = rec.Result()
    text = json.loads(result).get("text", "")

    if text:
        print(f"You said: {text}")
        return text
    else:
        print("Could not recognize speech.")
        return ""


# Handling user input
while True:
    choice = input("Enter 't' for text input or 's' for speech input: ").strip().lower()
    if choice == 's':
        text = get_speech_input()
        break
    elif choice == 't':
        text = input("Enter the text for analysis: ")
        break
    else:
        print("Invalid input. Please enter 't' for text input or 's' for speech input.")

if not text:
    print("No valid input received. Exiting...")
    exit()

# Process the text with SpaCy
doc = nlp(text)

# Sentence Classification
sentence_type = classify_sentence(doc)
print(f"\nSentence Type: {sentence_type}")

# POS Count
pos_counts = count_pos_tags(doc)
print("\nPOS Tag Counts:")
for pos, count in pos_counts.items():
    print(f"{pos}: {count}")

# Tokenization and Stopword Removal
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
word_counts = Counter(filtered_tokens)

print("\nTokens:", tokens)
print("\nFiltered Tokens (No Stopwords):", filtered_tokens)
print("\nWord Frequencies:", word_counts)

# Named Entity Recognition (NER)
print("\nNamed Entities in the Text:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Part-of-Speech Tags
print("\nPart-of-Speech Tags:")
for token in doc:
    print(f"{token.text}: {token.pos_}")

# Sentiment Analysis
sentiment, polarity = analyze_sentiment(text)
print(f"\nSentiment Analysis: {sentiment}")
print(f"Polarity Score: {polarity}")
