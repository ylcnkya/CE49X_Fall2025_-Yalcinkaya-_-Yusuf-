import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "aec_ai_analysis_tagged.jsonl"
BAR_CHART_FILE = "task4_bar_chart.png"
NETWORK_GRAPH_FILE = "task4_network_graph.png"
MATURITY_CSV = "ai_maturity_ranking.csv"

# Load the dataset
df = pd.read_json(INPUT_FILE, lines=True)

# 1. BAR CHART: Number of Articles per Civil Engineering Area
def plot_bar_chart(df):
    plt.figure(figsize=(10, 6))
    # Explode Civil_Tags and count occurrences
    civil_counts = df.explode('Civil_Tags')['Civil_Tags'].value_counts().reset_index()
    civil_counts.columns = ['Discipline', 'Count']
    
    if not civil_counts.empty:
        sns.barplot(data=civil_counts, x='Count', y='Discipline', hue='Discipline', palette='viridis', legend=False)
        plt.title('Total Number of Articles per Civil Engineering Discipline')
        plt.xlabel('Article Count')
        plt.ylabel('Civil Engineering Discipline')
        plt.tight_layout()
        plt.savefig(BAR_CHART_FILE)
        print(f"✅ Bar Chart saved as {BAR_CHART_FILE}")

# 2. NETWORK GRAPH: Relationships between Disciplines and AI Technologies
def plot_network_graph(df):
    plt.figure(figsize=(12, 10))
    G = nx.Graph()

    for _, row in df.iterrows():
        # Connect Civil Areas to AI Technologies if both exist
        if isinstance(row['Civil_Tags'], list) and isinstance(row['AI_Tags'], list):
            for civil in row['Civil_Tags']:
                for ai in row['AI_Tags']:
                    if G.has_edge(civil, ai):
                        G[civil][ai]['weight'] += 1
                    else:
                        G.add_edge(civil, ai, weight=1)

    if len(G.edges) > 0:
        pos = nx.spring_layout(G, k=0.6, seed=42)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        # Normalize edge widths for better visibility
        max_w = max(weights) if weights else 1
        edge_widths = [(w / max_w) * 5 for w in weights]

        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue", alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.4)

        plt.title("Network Analysis: Correlation between Civil Disciplines and AI Tech")
        plt.axis('off')
        plt.savefig(NETWORK_GRAPH_FILE)
        print(f"✅ Network Graph saved as {NETWORK_GRAPH_FILE}")
    else:
        print("⚠️ No AI-Civil connections found for Network Graph.")

# 3. WORD CLOUDS: Discipline-Specific Key Terms
def generate_word_clouds(df):
    all_civil_tags = df.explode('Civil_Tags')['Civil_Tags'].dropna().unique()
    
    for disc in all_civil_tags:
        subset = df[df['Civil_Tags'].apply(lambda x: isinstance(x, list) and disc in x)]
        # Combine clean_tokens for the specific discipline
        text = " ".join([" ".join(tokens) for tokens in subset['clean_tokens']])
        
        if len(text.strip()) > 10:
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f"Frequent Keywords: {disc}")
            plt.axis('off')
            file_name = f"wordcloud_{disc.lower().replace(' ', '_')}.png"
            plt.savefig(file_name)
            plt.close()
    print("✅ Word Clouds generated for all disciplines.")

# 4. AI MATURITY RANKING: Quantitative Synthesis
def calculate_ai_maturity(df):
    print("\n--- 🏆 CALCULATING AI MATURITY RANKING ---")
    
    df_exploded = df.explode('Civil_Tags').dropna(subset=['Civil_Tags'])
    
    results = []
    for discipline, group in df_exploded.groupby('Civil_Tags'):
        count = len(group)
        # Calculate AI Diversity: Average number of AI tags per article in this discipline
        ai_tags_lengths = [len(tags) for tags in group['AI_Tags'] if isinstance(tags, list)]
        avg_ai_diversity = np.mean(ai_tags_lengths) if ai_tags_lengths else 0
        
        results.append({
            'Discipline': discipline,
            'Article_Count': count,
            'Avg_AI_Diversity': round(avg_ai_diversity, 3),
            'Maturity_Score': round(count * avg_ai_diversity, 2)
        })
    
    maturity_df = pd.DataFrame(results).sort_values(by='Maturity_Score', ascending=False)
    print(maturity_df.to_string(index=False))
    
    # Save results
    maturity_df.to_csv(MATURITY_CSV, index=False)
    print(f"\n💾 Maturity ranking saved to {MATURITY_CSV}")
    return maturity_df

# --- EXECUTION ---
if __name__ == "__main__":
    plot_bar_chart(df)
    plot_network_graph(df)
    generate_word_clouds(df)
    calculate_ai_maturity(df)