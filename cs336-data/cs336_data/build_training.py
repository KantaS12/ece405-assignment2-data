import os
from fastwarc.warc import ArchiveIterator, WarcRecordType
from implementation import extract_text, language_identification, gopher_quality_filter

def process_positive_examples(warc_path: str, output_file: str, max_examples: int = 5000):
    """
    Extracts, filters, and formats high-quality Wikipedia-linked data.
    """
    print(f"Processing positive examples from {warc_path}...")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        # We use fastwarc to iterate through the scraped WARC file
        for record in ArchiveIterator(open(warc_path, 'rb'), record_types=WarcRecordType.response):
            if count >= max_examples:
                break
                
            try:
                html_bytes = record.reader.read()
                
                # Extract plain text using your Resiliparse function
                text = extract_text(html_bytes)
                if not text.strip():
                    continue

                # Filter out non-English text using FastText
                lang, conf = language_identification(text)
                if lang != "en" or conf < 0.8:  # Adjust threshold as needed
                    continue

                # Apply rule-based Gopher filters
                if not gopher_quality_filter(text):
                    continue

                # If it survived all your filters, format and save it!
                clean_text = text.replace("\n", " ").replace("\r", " ").strip()
                out_f.write(f"__label__hq {clean_text}\n")
                count += 1
                
                if count % 500 == 0:
                    print(f"Saved {count} positive examples...")
                    
            except Exception as e:
                continue

    print(f"Finished. Saved {count} high-quality examples.")

if __name__ == "__main__":
    warc_file = "/home/kantas/koa_scratch/ece405-assignment2-data/data/subsampled_positive_urls.warc.gz"
    output_txt = "/home/kantas/koa_scratch/ece405-assignment2-data/data/train.txt"
    
    # Run the processing pipeline
    process_positive_examples(warc_file, output_txt)