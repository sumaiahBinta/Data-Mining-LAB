import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Viva\archive(1)\tripadvisor_hotel_reviews.csv"
dataset = pd.read_csv(file_path)

# Extract reviews for searching
documents = dataset['Review'].values

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
    return results[:50]  # Top 50 results

# Function for Search Engine 2 (TF-IDF)
def search_engine_2(query, docs):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    query_vector = tfidf_vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-50:][::-1]  # Top 50 results
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
    top_indices = similarities.argsort()[-50:][::-1]  # Top 50 results
    return [f"Doc {i}" for i in top_indices]

# Define a sample query
query = "great service"


# Perform searches
results_1 = search_engine_1(query, processed_docs)
results_2 = search_engine_2(query, processed_docs)
results_3 = search_engine_3(query, processed_docs)

# Display results in table format
print("Query:", query)
print("\n** Table 1: Rank Position Calculation **")
print(f"{'Rank':<5}{'Doc appears in Search Engine 1':<35}{'Doc appears in Search Engine 2':<35}{'Doc appears in LSI':<20}")
for i in range(50):  # Top 50 results
    doc1 = results_1[i] if i < len(results_1) else "-"
    doc2 = results_2[i] if i < len(results_2) else "-"
    doc3 = results_3[i] if i < len(results_3) else "-"
    print(f"{i+1:<5}{doc1:<35}{doc2:<35}{doc3:<20}")

# Calculate metrics
common_docs = set(results_1) & set(results_2) & set(results_3)
rank_displacement = sum(abs(results_1.index(doc) - results_3.index(doc)) for doc in common_docs if doc in results_1 and doc in results_3)
uncommon_docs = set(results_1 + results_2 + results_3) - common_docs

print("\nNumber of Common Docs in Search Result:", len(common_docs))
print("Rank Displacement (Total):", rank_displacement)
print("Number of Uncommon Docs:", len(uncommon_docs))
