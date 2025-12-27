import os

# --- CONFIGURATION ---
MAIN_FILE = "aec_ai_corpus_unified.jsonl"          # The destination (Your big file)
NEW_FILE = "aec_diversity_pack_cleaned.jsonl"       # The source (The cleaned file)

def main():
    print(f"🔄 Appending '{NEW_FILE}' to '{MAIN_FILE}'...")
    
    # Check if files exist
    if not os.path.exists(MAIN_FILE):
        print(f"❌ Error: Main file '{MAIN_FILE}' not found.")
        return
    if not os.path.exists(NEW_FILE):
        print(f"❌ Error: New file '{NEW_FILE}' not found.")
        return

    count = 0
    
    # Open Main file in APPEND mode ('a') and New file in READ mode ('r')
    with open(MAIN_FILE, 'a', encoding='utf-8') as outfile:
        with open(NEW_FILE, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip(): # Only write non-empty lines
                    outfile.write(line)
                    count += 1

    print(f"✅ Successfully added {count} new articles.")
    print(f"📁 Updated File: {MAIN_FILE}")

if __name__ == "__main__":
    main()