import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Path to your dataset
file_path = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Lab 3\archive(1)\tripadvisor_hotel_reviews.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Let's inspect the first few rows of the dataset to understand its structure
print(df.head())

# Assuming the review text is in a column named 'Review'
documents = df['Review'].tolist()

# Initialize CountVectorizer to extract vocabulary and create term-document matrix
vectorizer = CountVectorizer(stop_words='english')  # remove common stopwords
X = vectorizer.fit_transform(documents)  # Sparse matrix

# Create a sparse DataFrame from the term-document matrix
term_doc_matrix = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names_out())

# Extract vocabulary terms and their frequency across documents (summing across columns)
term_freq = term_doc_matrix.sum().sort_values(ascending=False)

# Get the top 100 terms based on frequency
top_100_terms = term_freq.head(100)

# Partial term-document matrix for top 100 terms
partial_matrix = term_doc_matrix[top_100_terms.index]

# Report generation
report = {
    'Number of documents (samples)': X.shape[0],
    'Total categories': 'Hotel Review',  # Added hotel review as the category
    'Total number of unique terms': X.shape[1],
    'Top 100 terms': top_100_terms.index.tolist(),
    'Partial Term-Document Matrix (for top 100 terms)': partial_matrix
}

# Update paths to save the report and matrix to your desired location
report_filename = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Lab 3\report.txt"
top_100_matrix_filename = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Lab 3\top_100_terms_matrix.csv"

# Save the partial term-document matrix as a CSV file
partial_matrix.to_csv(top_100_matrix_filename)

# Generate a text-based report and save to the specified path
with open(report_filename, 'w') as f:
    f.write(f"Number of documents (samples): {report['Number of documents (samples)']}\n")
    f.write(f"Total categories: {report['Total categories']}\n")
    f.write(f"Total number of unique terms: {report['Total number of unique terms']}\n")
    f.write("\nTop 100 terms:\n")
    for term in report['Top 100 terms']:
        f.write(f"{term}\n")
    f.write("\nPartial Term-Document Matrix (first 5 rows shown):\n")
    f.write(partial_matrix.head().to_string())

print(f"Report saved to {report_filename}")
print(f"Partial Term-Document Matrix saved to {top_100_matrix_filename}")
