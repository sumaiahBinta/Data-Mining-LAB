import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Viva\archive(1)\tripadvisor_hotel_reviews.csv"
dataset = pd.read_csv(file_path)

# Extract reviews for searching
documents = dataset['Review'].dropna().tolist()  # Ensure no null values
print(f"Loaded {len(documents)} reviews from the dataset.")

# Preprocess text (basic preprocessing)
def preprocess_text(text):
    return text.lower()

processed_docs = [preprocess_text(doc) for doc in documents]

# Function for Search Engine 1 (Keyword Search)
def search_engine_1(query, docs):
    query = preprocess_text(query)
    results = []
    for idx, doc in enumerate(docs):
        if query in doc:
            results.append(f"Doc {idx}")
    return results[:100]  # Top 100 results

# Function for Search Engine 2 (TF-IDF)
def search_engine_2(query, docs):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    query_vector = tfidf_vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-100:][::-1]  # Top 100 results
    return [f"Doc {i}" for i in top_indices]

# Function for Search Engine 3 (LSI)
def search_engine_3(query, docs, f_number=10):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    
    # Apply LSI
    svd = TruncatedSVD(n_components=f_number, random_state=42)
    lsi_matrix = svd.fit_transform(tfidf_matrix)
    query_vector = tfidf_vectorizer.transform([query])
    query_lsi = svd.transform(query_vector)
    
    similarities = cosine_similarity(query_lsi, lsi_matrix).flatten()
    top_indices = similarities.argsort()[-100:][::-1]  # Top 100 results
    return [f"Doc {i}" for i in top_indices]

# Define queries for search (example real-world queries)
queries = [
    "great service",
    "clean rooms",
    "excellent food",
    "poor hygiene",
    "friendly staff",
    "location and amenities",
    "value for money",
    "quick check-in",
    "slow service",
    "amazing experience"
]

# Display queries before results
print("\n# Queries for Search (Real-World Examples):")
for i, query in enumerate(queries, start=1):
    print(f"{i}. {query}")

# Perform searches and collect results
results_table = []

for query in queries:
    results_1 = search_engine_1(query, processed_docs)
    results_2 = search_engine_2(query, processed_docs)
    results_3 = search_engine_3(query, processed_docs)

    # Append results to the table
    results_table.append({
        "Query": query,
        "Search Engine 1": results_1,
        "Search Engine 2": results_2,
        "Search Engine 3": results_3
    })

# Compare results for the first query (as an example for the output table)
query_results = results_table[0]  # Taking the first query for comparison

# Find common, uncommon documents, and calculate rank displacement
common_docs = set(query_results["Search Engine 1"]) & set(query_results["Search Engine 2"]) & set(query_results["Search Engine 3"])
uncommon_docs = set(query_results["Search Engine 1"] + query_results["Search Engine 2"] + query_results["Search Engine 3"]) - common_docs

rank_displacement = 0
for doc in common_docs:
    rank_1 = query_results["Search Engine 1"].index(doc) if doc in query_results["Search Engine 1"] else -1
    rank_3 = query_results["Search Engine 3"].index(doc) if doc in query_results["Search Engine 3"] else -1
    rank_displacement += abs(rank_1 - rank_3)

# Display comparison table
print("\n** Table 1: Rank Position Calculation **")
print(f"{'Rank':<5}{'Doc appears in Search Engine 1':<35}{'Doc appears in Search Engine 2':<35}{'Doc appears in LSI':<20}")
for i in range(100):  # Top 100 results
    doc1 = query_results["Search Engine 1"][i] if i < len(query_results["Search Engine 1"]) else "-"
    doc2 = query_results["Search Engine 2"][i] if i < len(query_results["Search Engine 2"]) else "-"
    doc3 = query_results["Search Engine 3"][i] if i < len(query_results["Search Engine 3"]) else "-"
    print(f"{i+1:<5}{doc1:<35}{doc2:<35}{doc3:<20}")

# Display metrics
print("\nNumber of Common Docs in Search Result:", len(common_docs))
print("Rank Displacement (Total):", rank_displacement)
print("Number of Uncommon Docs:", len(uncommon_docs))
