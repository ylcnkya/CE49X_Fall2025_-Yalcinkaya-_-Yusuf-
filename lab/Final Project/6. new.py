import json
import os

# --- CONFIGURATION ---
INPUT_FILE = "aec_ai_corpus_complete.jsonl"
OUTPUT_FILE = "aec_ai_corpus_unified.jsonl"  # Saving to a new file is safer

def main():
    print(f"🧹 Removing 'source_type' from '{INPUT_FILE}'...")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: Input file not found.")
        return

    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                
                # REMOVE source_type
                if 'source_type' in data:
                    del data['source_type']
                
                # Write the clean line
                outfile.write(json.dumps(data) + "\n")
                count += 1
            except json.JSONDecodeError:
                continue

    print(f"✅ Processed {count} articles.")
    print(f"📁 Clean file saved as: {OUTPUT_FILE}")
    
    # Optional: Ask user if they want to overwrite the original
    # os.replace(OUTPUT_FILE, INPUT_FILE) 

if __name__ == "__main__":
    main()