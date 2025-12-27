import json

# --- CONFIGURATION ---
INPUT_FILE = "aec_diversity_pack_with_source.jsonl"
OUTPUT_FILE = "aec_diversity_pack_cleaned.jsonl"

def main():
    print(f"🧹 Cleaning '{INPUT_FILE}'...")
    
    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                
                # REMOVE THE CATEGORY FIELD
                if 'category' in data:
                    del data['category']
                
                # Also remove 'assigned_category' if it exists, just to be safe
                if 'assigned_category' in data:
                    del data['assigned_category']
                
                outfile.write(json.dumps(data) + "\n")
                count += 1
            except json.JSONDecodeError:
                continue

    print(f"✅ Removed category tags from {count} articles.")
    print(f"📁 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()