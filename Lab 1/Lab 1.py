import pandas as pd
import os

# Load the dataset from your path
file_path = r"E:\4-2 term\DM Lab\archive\Groceries_dataset.csv"
df = pd.read_csv(file_path)

# ------------------------
print(f"------------------------")
# 1. Domain of the dataset
domain = "Retail and Grocery Transactions"
print(f"1. Domain: {domain}")
print(f"------------------------")

# 2. Number of samples (total transactions)
num_samples = len(df)
print(f"2. Number of samples: {num_samples}")
print(f"------------------------")

# 3. Number of unique categories (items)
unique_items = df['itemDescription'].nunique()
print(f"3. Number of unique categories (items): {unique_items}")
print(f"------------------------")

# 4. Dataset size in KB
dataset_size_bytes = os.path.getsize(file_path)  # Size in bytes
dataset_size_kb = dataset_size_bytes / 1024  # Convert to KB
print(f"4. Dataset size: {dataset_size_kb:.2f} KB")
print(f"------------------------")

# 5. Unique items (Optional: see distinct items)
unique_items_list = df['itemDescription'].unique()
print(f"5. Unique items: {unique_items_list}")  # Display only first 10 unique items for readability
print(f"------------------------")
