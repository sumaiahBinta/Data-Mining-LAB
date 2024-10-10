from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # Split items by comma and strip whitespace
    dataset = [row.split(',') for row in df['Product']]
    # Strip whitespace from each item
    dataset = [[item.strip() for item in transaction] for transaction in dataset]
    return dataset

# Step 1: Generate L1 (frequent 1-itemsets)
def generate_L1(item_support, min_support):
    return {frozenset([item]): support for item, support in item_support.items() if support >= min_support}

# Step 2: Generate candidate k-itemsets (Ck) from Lk-1
def generate_candidates(Lk_minus_1, k):
    candidates = set()
    Lk_list = list(Lk_minus_1)

    for i in range(len(Lk_list)):
        for j in range(i + 1, len(Lk_list)):
            l1 = list(Lk_list[i])
            l2 = list(Lk_list[j])
            # Join condition
            if l1[:k-2] == l2[:k-2]:
                candidate = frozenset(Lk_list[i] | Lk_list[j])
                candidates.add(candidate)

    return candidates

# Step 3: Prune step
def prune_candidates(candidates, Lk_minus_1):
    pruned_candidates = set()

    for candidate in candidates:
        is_frequent = True
        for subset in map(frozenset, [set(candidate) - {item} for item in candidate]):
            if subset not in Lk_minus_1:
                is_frequent = False
                break

        if is_frequent:
            pruned_candidates.add(candidate)

    return pruned_candidates

# Step 4: Count support for each candidate k-itemset
def count_support(dataset, candidates):
    support_count = defaultdict(int)

    for transaction in dataset:
        transaction_set = frozenset(transaction)
        for candidate in candidates:
            if candidate.issubset(transaction_set):
                support_count[candidate] += 1

    return support_count

# Step 5: Generate Lk from Ck
def generate_Lk(candidates, support_count, min_support):
    Lk = {candidate: support for candidate, support in support_count.items() if support >= min_support}
    return Lk

# Full Apriori Algorithm
def apriori_algorithm(dataset, min_support):
    # Step 1: Generate item support
    item_support = defaultdict(int)
    for transaction in dataset:
        for item in transaction:
            item_support[item] += 1

    # Generate L1
    L1 = generate_L1(item_support, min_support)
    print("L1 (Frequent 1-itemsets):")
    for itemset, support in L1.items():
        print(f"Itemset: {', '.join(list(itemset))}, Support: {support}")
    print()

    all_frequent_itemsets = dict(L1)

    Lk = L1
    k = 2

    while Lk:
        candidates = generate_candidates(Lk.keys(), k)
        candidates = prune_candidates(candidates, Lk.keys())
        support_count = count_support(dataset, candidates)
        Lk = generate_Lk(candidates, support_count, min_support)
        if Lk:
            print(f"L{k} (Frequent {k}-itemsets):")
            for itemset, support in Lk.items():
                print(f"Itemset: {', '.join(list(itemset))}, Support: {support}")
            print()
        all_frequent_itemsets.update(Lk)
        k += 1

    return all_frequent_itemsets

# Run Apriori algorithm for different min_support values
def run_apriori_for_min_supports(dataset, min_support_range):
    support_results = []

    for min_support in min_support_range:
        print(f"\nRunning Apriori with min_support = {min_support}")
        frequent_itemsets = apriori_algorithm(dataset, min_support)
        num_itemsets = len(frequent_itemsets)
        support_results.append((min_support, num_itemsets))

    return support_results

# Visualize the effect of varying minimum support
def plot_support_vs_itemsets(support_results):
    min_supports, itemset_counts = zip(*support_results)
    plt.figure(figsize=(10, 6))
    plt.plot(min_supports, itemset_counts, marker='o')
    plt.title('Number of Frequent Itemsets vs Min Support')
    plt.xlabel('Min Support')
    plt.ylabel('Number of Frequent Itemsets')
    plt.grid(True)
    plt.show()

# File path to your CSV dataset
file_path = r"archive/groceries_dataset.csv"

# Ensure the dataset is loaded
dataset = load_dataset(file_path)
#print(f"Sample dataset: {dataset[:5]}")  # Printing first 5 rows for verification

# Define a range of minimum support values (adjust as needed)
min_support_range = range(1, 30, 2)

# Run Apriori algorithm for different min_support values
support_results = run_apriori_for_min_supports(dataset, min_support_range)

# Plot the results
plot_support_vs_itemsets(support_results)
