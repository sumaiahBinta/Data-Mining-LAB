import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the dataset
file_path = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Lab 5\archive(1)\tripadvisor_hotel_reviews.csv"
data = pd.read_csv(file_path)

# Extract reviews
reviews = data['Review']

# Step 1: Calculate the Term-Document Matrix using TF-IDF
def compute_tfidf(reviews, max_features=10):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(reviews)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_matrix, vectorizer, tfidf_df

# Step 2: Calculate the Term-Term Correlation Matrix
def compute_term_correlation(tfidf_df):
    correlation_matrix = np.corrcoef(tfidf_df.T)
    terms = tfidf_df.columns
    correlation_df = pd.DataFrame(correlation_matrix, index=terms, columns=terms)
    return correlation_df

# Step 3: Define the Search Engine Function using TF-IDF and Summation-Based Ranking
def search_engine(query, vectorizer, tfidf_matrix, data, top_n=50):
    query_vec = vectorizer.transform([query.lower()])
    query_terms = vectorizer.get_feature_names_out()
    
    # Find indices of query terms in TF-IDF matrix if they exist
    query_term_indices = [i for i, term in enumerate(query_terms) if term in query.split()]
    
    if not query_term_indices:
        return [], []
    
    # Calculate the summation of TF-IDF weights for ranking
    tfidf_sums = tfidf_matrix.toarray()[:, query_term_indices].sum(axis=1)
    top_indices = tfidf_sums.argsort()[-top_n:][::-1]
    
    # Gather top 10 results for display with their total weights
    top_10_results = [{"Review Index": idx, "Total Weight": tfidf_sums[idx]} for idx in top_indices[:10]]
    
    return top_10_results, top_indices

# Step 4: Compare Displacement of Rankings
def calculate_displacement(previous_ranking, new_ranking):
    rank_displacements = []
    
    for new_rank, review_index in enumerate(new_ranking[:50]):
        if review_index in previous_ranking:
            previous_rank = previous_ranking.index(review_index)
            displacement = abs(previous_rank - new_rank)
            rank_displacements.append({
                "Review Index": review_index,
                "Previous Rank": previous_rank + 1,
                "New Rank": new_rank + 1,
                "Displacement": displacement
            })
    
    displacement_df = pd.DataFrame(rank_displacements)
    return displacement_df

# Execute Steps

# 1. Calculate TF-IDF term-document matrix
tfidf_matrix, vectorizer, tfidf_df = compute_tfidf(reviews)

# Show top 10 terms from the TF-IDF term-document matrix
top_10_terms_df = tfidf_df.sum().sort_values(ascending=False).head(10)
print("Top 10 Terms in Term-Document Matrix (by TF-IDF weight):\n", top_10_terms_df)

# 2. Calculate term-term correlation matrix
correlation_df = compute_term_correlation(tfidf_df)

# Show top 10 terms in Term-Term Correlation Matrix
top_10_correlation_df = correlation_df[top_10_terms_df.index].loc[top_10_terms_df.index]
print("\nTop 10 Terms in Term-Term Correlation Matrix:\n", top_10_correlation_df)

# 3. Example Search Query
query = "excellent service and clean rooms"

# Simulate a previous ranking (randomly generated for example purposes)
np.random.seed(42)
previous_ranking = np.random.permutation(len(reviews)).tolist()

# Perform the new search and get new ranking with top 10 results
top_10_results, new_ranking = search_engine(query, vectorizer, tfidf_matrix, data)

# Display the top 10 results and their total TF-IDF weights
print("\nTop 10 Search Results for Query '{}':".format(query))
for result in top_10_results:
    print(f"Review Index: {result['Review Index']}, Total Weight: {result['Total Weight']}")

# 4. Calculate displacement of ranks
displacement_df = calculate_displacement(previous_ranking, new_ranking)

# Display rank displacement table for top 50 results
print("\nDisplacement Table (Top 50 Results):\n", displacement_df)
