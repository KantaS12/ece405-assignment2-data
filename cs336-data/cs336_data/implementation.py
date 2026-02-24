from resiliparse.parse.encoding import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text

import os
_FASTTEXT_MODEL_LANGUAGE = None
_FASTTEXT_MODEL_NSFW = None
_FASTTEXT_MODEL_TOXIC_SPEECH = None
_FASTTEXT_MODEL_QUALITY = None


def extract_text(html: bytes) -> str:
    # Decode byte into unicode string
    try:
        html_str = html.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        encoding_type = detect_encoding(html)
        html_str = html.decode(encoding_type, errors="ignore")

    # Extract from unicode to text
    tree = HTMLTree.parse(html_str)
    text = extract_plain_text(tree)

    return text


def language_identification(text: str) -> tuple[str, float]:
    import fasttext

    global _FASTTEXT_MODEL_LANGUAGE
    if _FASTTEXT_MODEL_LANGUAGE is None:
        model_path = os.path.join(os.path.dirname(__file__), "var", "lid.176.bin")
        _FASTTEXT_MODEL_LANGUAGE = fasttext.load_model(model_path)

    text = text.replace("\n", " ")

    labels, prob = _FASTTEXT_MODEL_LANGUAGE.predict(text)

    language = labels[0].replace("__label__", "")
    
    confidence = prob[0]

    return (language, confidence)


# Mask Pii
import re

def mask_emails(text: str) -> tuple[str, int]:
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)

    count = len(re.findall(email_pattern, text))

    return masked_text, count

def mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'

    masked_text = re.sub(phone_pattern, '|||PHONE_NUMBER|||', text)

    count = len(re.findall(phone_pattern, text))

    return masked_text, count

def mask_IP_addresses(text: str) -> tuple[str, int]:
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'

    masked_text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)

    count = len(re.findall(ip_pattern, text))

    return masked_text, count


def classify_NSFW(text: str) -> tuple[str, float]:
    import fasttext

    global _FASTTEXT_MODEL_NSFW

    if _FASTTEXT_MODEL_NSFW is None:
        model_path = os.path.join(os.path.dirname(__file__), "var", "jigsaw_fasttext_bigrams_nsfw_final.bin")
        _FASTTEXT_MODEL_NSFW = fasttext.load_model(model_path)


    text = text.replace("\n", " ")

    labels, prob = _FASTTEXT_MODEL_NSFW.predict(text)

    nsfw_label = labels[0].replace("__label__", "")
    nsfw_confidence = prob[0]

    return nsfw_label, nsfw_confidence


def classify_toxic_speech(text: str) -> tuple[str, float]:
    import fasttext

    global _FASTTEXT_MODEL_TOXIC_SPEECH

    if _FASTTEXT_MODEL_TOXIC_SPEECH is None:
        model_path = os.path.join(os.path.dirname(__file__), "var", "jigsaw_fasttext_bigrams_hatespeech_final.bin")
        _FASTTEXT_MODEL_TOXIC_SPEECH = fasttext.load_model(model_path)

    text = text.replace("\n", " ")

    labels, prob = _FASTTEXT_MODEL_TOXIC_SPEECH.predict(text)

    toxic_label = labels[0].replace("__label__", "")
    toxic_confidence = prob[0]

    return toxic_label, toxic_confidence


def gopher_quality_filter(text: str) -> bool:
    import nltk

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Tokenize the text into sentences
    sentences = nltk.word_tokenize(text)

    # Remove if contains less than 50 or more than 100,000 words
    if len(sentences) < 50 or len(sentences) > 100000:
        return False
    
    # Have a mean word length outside the range 3 to 10 characters
    word_lengths = [len(word) for word in sentences]
    mean_word_length = sum(word_lengths) / len(word_lengths)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Have more than 30% of lines ending with an ellipsis ("...")
    lines = text.splitlines()
    if len(lines) > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False
    
    # Contains less than 80% of words with at least one alphabetic character
    alphabetic_words = sum(1 for word in sentences if any(c.isalpha() for c in word))
    if len(sentences) > 0 and alphabetic_words / len(sentences) < 0.8:
        return False

    return True


def quality_classifier(text: str) -> tuple[str, float]:
    import fasttext

    global _FASTTEXT_MODEL_QUALITY

    if _FASTTEXT_MODEL_QUALITY is None:
        model_path = os.path.join(os.path.dirname(__file__), "var", "quality_classifier.bin")
        _FASTTEXT_MODEL_QUALITY = fasttext.load_model(model_path)

    text = text.replace("\n", " ")

    labels, prob = _FASTTEXT_MODEL_QUALITY.predict(text)

    quality_label = labels[0].replace("__label__", "")
    quality_confidence = prob[0]

    return quality_label, quality_confidence