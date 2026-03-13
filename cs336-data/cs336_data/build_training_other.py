import os
from fastwarc.warc import ArchiveIterator, WarcRecordType
from implementation import extract_text, language_identification, gopher_quality_filter

def process_negative_examples(warc_path: str, output_file: str, max_examples: int = 5000):
    """
    Extracts, filters, and formats LOW-quality web data.
    """
    print(f"Processing negative examples from {warc_path}...")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        # We use fastwarc to iterate through the scraped WARC file
        for record in ArchiveIterator(open(warc_path, 'rb'), record_types=WarcRecordType.response):
            if count >= max_examples:
                break
                
            try:
                html_bytes = record.reader.read()
                
                # Extract plain text
                text = extract_text(html_bytes)
                if not text.strip():
                    continue

                lang, conf = language_identification(text)
                if lang != "en" or conf < 0.8:
                    continue

                clean_text = text.replace("\n", " ").replace("\r", " ").strip()
                out_f.write(f"__label__lq {clean_text}\n")
                count += 1
                
                if count % 500 == 0:
                    print(f"Saved {count} negative examples...")
                    
            except Exception as e:
                continue

    print(f"Finished. Saved {count} low-quality examples.")

if __name__ == "__main__":
    warc_file = "/home/kantas/koa_scratch/ece405-assignment2-data/data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    output_txt = "/home/kantas/koa_scratch/ece405-assignment2-data/data/train_negative.txt"
    
    # Run the processing pipeline
    process_negative_examples(warc_file, output_txt)