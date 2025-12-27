import requests
import trafilatura
import xml.etree.ElementTree as ET
import time
import random
import json
import logging
from urllib.parse import urlparse

# --- CONFIGURATION ---
# We want ~250 new articles (50 per category) from OTHER sources
TOTAL_TARGET = 250
QUOTA_PER_CATEGORY = 50 
OUTPUT_FILE = "aec_diversity_pack.jsonl" # New file!
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# --- SOURCE DEFINITIONS (AECMAG REMOVED) ---
SOURCES = {
    "specialized_aec": [
        "https://www.bimplus.co.uk/sitemap.xml",
        # "https://www.aecmag.com/sitemap_index.xml",  <-- REMOVED
        "https://www.constructiondive.com/feeds/news/",
        "https://www.enr.com/rss/headlines",
        "https://globalconstructionreview.com/feed/",
        "https://www.geospatialworld.net/sitemap_index.xml"
    ],
    "general_tech": [
        "https://techcrunch.com/feed/", 
        "https://venturebeat.com/feed/",
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://www.theverge.com/rss/index.xml"
    ]
}

CIVIL_CATEGORIES = {
    "Structural": ["structural", "steel", "concrete", "seismic", "beam", "column", "bridge", "load bearing"],
    "Geotechnical": ["geotechnical", "soil", "foundation", "tunnel", "excavation", "underground", "earthwork"],
    "Transportation": ["transportation", "traffic", "road", "highway", "rail", "transit", "autonomous vehicle"],
    "Construction_Mgmt": ["schedule", "scheduling", "safety", "cost", "estimation", "site management", "procurement"],
    "Environmental": ["sustainability", "waste", "green building", "carbon", "energy efficiency", "leed", "emissions"]
}

# --- HELPER FUNCTIONS (Same as before) ---
def get_links_from_rss(feed_url):
    links = []
    try:
        resp = requests.get(feed_url, headers={'User-Agent': USER_AGENT}, timeout=5)
        root = ET.fromstring(resp.content)
        for item in root.findall('.//item'):
            link = item.find('link').text
            if link: links.append(link)
    except: pass
    return links

def get_links_from_sitemap(sitemap_url):
    links = []
    try:
        resp = requests.get(sitemap_url, headers={'User-Agent': USER_AGENT}, timeout=5)
        root = ET.fromstring(resp.content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemaps = root.findall('ns:sitemap/ns:loc', ns)
        if sitemaps:
            for sm in sitemaps[:3]: 
                links.extend(get_links_from_sitemap(sm.text))
        else:
            urls = root.findall('ns:url/ns:loc', ns)
            for url in urls:
                links.append(url.text)
    except: pass
    return links

category_counts = {k: 0 for k in CIVIL_CATEGORIES.keys()}

def classify_and_assign(text):
    content = text.lower()
    AI_TERMS = ["artificial intelligence", " ai ", "machine learning", "computer vision", "generative", "neural network", "robot", "automation", "predictive"]
    
    if not any(t in content for t in AI_TERMS): return None

    valid_cats = {}
    for cat, terms in CIVIL_CATEGORIES.items():
        count = sum(1 for term in terms if term in content)
        if count > 0: valid_cats[cat] = count
            
    if not valid_cats: return None

    # Pick best fit that isn't full
    sorted_cats = sorted(valid_cats.items(), key=lambda x: x[1], reverse=True)
    for cat, score in sorted_cats:
        if category_counts[cat] < QUOTA_PER_CATEGORY:
            return cat
    return None

def run_harvester():
    all_links = []
    print("--- 1. GATHERING DIVERSITY LINKS ---")
    
    for source in SOURCES["specialized_aec"] + SOURCES["general_tech"]:
        if "feed" in source or "rss" in source:
            all_links.extend(get_links_from_rss(source))
        else:
            all_links.extend(get_links_from_sitemap(source))

    unique_links = list(set(all_links))
    random.shuffle(unique_links)
    
    print(f"✅ Processing {len(unique_links)} URLs. Goal: {QUOTA_PER_CATEGORY} NON-AECMAG articles per category.")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for url in unique_links:
            if "aecmag.com" in url: continue # Double Check Safety
            
            if sum(category_counts.values()) >= TOTAL_TARGET: break
                
            try:
                time.sleep(random.uniform(0.5, 1.0))
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    result = trafilatura.extract(downloaded, output_format="json", with_metadata=True, include_comments=False)
                    if result:
                        doc = json.loads(result)
                        full_text = doc.get('text', '')
                        cat = classify_and_assign(full_text)
                        
                        if cat:
                            entry = {"url": url, "title": doc.get('title'), "date": doc.get('date'), "category": cat, "text": full_text}
                            f.write(json.dumps(entry) + "\n")
                            category_counts[cat] += 1
                            print(f"✅ [{cat}] Saved: {url[:40]}...")
            except: continue

    print("🎉 Diversity Collection Complete!")

if __name__ == "__main__":
    run_harvester()