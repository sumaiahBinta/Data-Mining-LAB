import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the dataset
file_path = "E:/4-2 term/DM Lab/Data-Mining-Lab/Lab 4/archive(1)/tripadvisor_hotel_reviews.csv"
df = pd.read_csv(file_path)

# Extract the review texts from the dataset
documents = df['Review'].tolist()

# Initialize CountVectorizer to create the term-document matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Create a DataFrame for the term-document matrix
term_doc_matrix = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names_out())

# Function to process the query and rank search results based on term frequencies
def search_query(query, top_n=10):
    # Tokenize the query and vectorize it
    query_vector = vectorizer.transform([query]).toarray()
    query_terms = vectorizer.get_feature_names_out()[np.where(query_vector[0] > 0)]
    
    # Find the total frequency of query terms in each document
    doc_scores = term_doc_matrix[query_terms].sum(axis=1)
    
    # Get the top N results sorted by the score
    top_n_indices = doc_scores.nlargest(top_n).index
    top_n_scores = doc_scores.nlargest(top_n)
    
    # Store the results to display
    results = []
    for i, idx in enumerate(top_n_indices):
        results.append({
            'rank': i + 1,
            'review': df['Review'][idx],
            'total_weight': top_n_scores[idx]
        })
    return results

# Example search query
query = "clean bathroom"
search_results = search_query(query)

# Display the top 10 search results
for result in search_results:
    print(f"Rank {result['rank']}:")
    print(f"Review: {result['review']}")
    print("-" * 80)
    print(f"Total Weight: {result['total_weight']}")
    print("-" * 150)
