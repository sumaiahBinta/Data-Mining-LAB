import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the dataset with your specified file path
file_path = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Lab 5\archive(1)\tripadvisor_hotel_reviews.csv"
data = pd.read_csv(file_path)

# Extract reviews
reviews = data['Review']

# Step 1: Compute the Full TF-IDF Matrix
def compute_tfidf(reviews, max_features=10):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(reviews)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_matrix, vectorizer, tfidf_df

# Step 2: Calculate Full Term-Term Correlation Matrix
def compute_term_correlation(tfidf_df):
    correlation_matrix = np.corrcoef(tfidf_df.T)
    terms = tfidf_df.columns
    correlation_df = pd.DataFrame(correlation_matrix, index=terms, columns=terms)
    return correlation_df

# Step 3: Define the Search Engine Function with Summation-Based Ranking
def search_engine(query, vectorizer, tfidf_matrix, data, top_n=50):
    query_vec = vectorizer.transform([query.lower()])
    query_terms = vectorizer.get_feature_names_out()
    
    # Find indices of query terms in TF-IDF matrix if they exist
    query_term_indices = [i for i, term in enumerate(query_terms) if term in query.split()]
    
    if not query_term_indices:
        return [], []
    
    tfidf_sums = tfidf_matrix.toarray()[:, query_term_indices].sum(axis=1)
    top_indices = tfidf_sums.argsort()[-top_n:][::-1]
    return top_indices[:10], top_indices

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
tfidf_matrix, vectorizer, tfidf_df = compute_tfidf(reviews)
correlation_df = compute_term_correlation(tfidf_df)

# Example Search Query
query = "excellent service and clean rooms"

# Simulate a previous ranking (randomly generated for example purposes)
np.random.seed(42)
previous_ranking = np.random.permutation(len(reviews)).tolist()

# Perform the new search and get new ranking
top_10_results, new_ranking = search_engine(query, vectorizer, tfidf_matrix, data)

# Calculate displacement of ranks
displacement_df = calculate_displacement(previous_ranking, new_ranking)

# Display output
print("Top 10 Results from Search:\n", top_10_results)
print("\nDisplacement Table (Top 50 Results):\n", displacement_df)
