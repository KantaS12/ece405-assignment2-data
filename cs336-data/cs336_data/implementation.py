from resiliparse.parse.encoding import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text

import os
import mmh3
from collections import Counter

import re
import unicodedata
import random
from collections import defaultdict
from itertools import combinations

_FASTTEXT_MODEL_LANGUAGE = None
_FASTTEXT_MODEL_NSFW = None
_FASTTEXT_MODEL_TOXIC_SPEECH = None
_FASTTEXT_MODEL_QUALITY = None


def extract_text(html: bytes) -> str:
    # Decode byte into unicode string
    try:
        html_str = html.decode("utf-8", errors="strict")
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

    clean_text = text.replace("\n", " ").replace("\r", " ").strip()

    labels, prob = _FASTTEXT_MODEL_QUALITY.predict(clean_text)

    quality_label = labels[0].replace("__label__", "")
    quality_confidence = float(prob[0])

    return quality_label, quality_confidence


def exact_deduplication(input_path: os.PathLike, output_path: os.PathLike):
    # Count frequency of each line in the corpus, using hash

    # Rewrite input file to output directory with same name, deduplicate 
    # content by removing lines that occur more than once in the set of input files

   os.makedirs(output_path, exist_ok=True)

   hash_counts = Counter()

   for filepath in input_path:
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_hash = mmh3.hash(line.encode('utf-8'))
            hash_counts[line_hash] += 1


    for filepath in input_path:
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_path, filename)

        with open(filepath, 'r', encoding='utf-8') as fin, \
             open(output_filepath, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line_hash = mmh3.hash(line.encode('utf-8'))
                
                if hash_counts[line_hash] == 1:
                    fout.write(line)


def minhash_deduplication(input_path: os.PathLike, num_hashes: int, num_bands: int, ngrams: int, jaccard_threshold: float, output_path: os.PathLike):

    os.makedirs(output_path, exist_ok=True)
    r = num_hashes // num_bands
    
    lsh_buckets = defaultdict(list)
    doc_ngrams = {}
    
    # Inline helper
    parent = {}
    def find(i):
        if parent.setdefault(i, i) != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Normalization, N-Grams, MinHashing, and LSH
    for filepath in input_path:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read the entire file as one single document
            raw_text = f.read()
            
            # Inline Normalization
            text = raw_text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Inline N-Grams
            words = text.split()
            if len(words) >= ngrams:
                doc_ngram_set = set(tuple(words[i:i+ngrams]) for i in range(len(words) - ngrams + 1))
            else:
                doc_ngram_set = set()
            
            # The filepath itself is our unique document ID now
            doc_ngrams[filepath] = doc_ngram_set
            
            # Inline MinHash Signatures
            signature = []
            for seed in range(num_hashes):
                if not doc_ngram_set:
                    signature.append(0)
                else:
                    min_h = min(mmh3.hash(" ".join(g).encode('utf-8'), seed) for g in doc_ngram_set)
                    signature.append(min_h)
                    
            # LSH Banding
            for band_idx in range(num_bands):
                band_tuple = tuple(signature[band_idx * r : (band_idx + 1) * r])
                lsh_buckets[(band_idx, band_tuple)].append(filepath)

    # Candidate Clustering and verification
    for doc_ids in lsh_buckets.values():
        if len(doc_ids) > 1:
            for doc_a, doc_b in combinations(doc_ids, 2):
                if find(doc_a) == find(doc_b):
                    continue
                
                # True Jaccard Verification
                set_a, set_b = doc_ngrams[doc_a], doc_ngrams[doc_b]
                if not set_a and not set_b:
                    sim = 0.0
                else:
                    sim = len(set_a & set_b) / len(set_a | set_b)
                    
                if sim >= jaccard_threshold:
                    union(doc_a, doc_b)
                    
    # Group connected components
    clusters = defaultdict(list)
    for filepath in input_path:
        clusters[find(filepath)].append(filepath)
        
    # Randomly select exactly one document to keep from each cluster
    files_to_keep = set()
    for cluster_docs in clusters.values():
        files_to_keep.add(random.choice(cluster_docs))

    # Write output paths
    for filepath in files_to_keep:
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(output_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as fin, \
             open(out_filepath, 'w', encoding='utf-8') as fout:
            # Copy Entire file
            fout.write(fin.read())