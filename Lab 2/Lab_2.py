import pandas as pd
from collections import defaultdict
from itertools import combinations

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

# Step 2: Generate candidate k-itemsets 
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
    print(f"L1 (Frequent 1-itemsets): {L1}\n")

    all_frequent_itemsets = dict(L1)

    Lk = L1
    k = 2

    while Lk:
        candidates = generate_candidates(Lk.keys(), k)
        candidates = prune_candidates(candidates, Lk.keys())
        support_count = count_support(dataset, candidates)
        Lk = generate_Lk(candidates, support_count, min_support)
        all_frequent_itemsets.update(Lk)
        print(f"L{k} (Frequent {k}-itemsets): {Lk}\n")
        k += 1

    return all_frequent_itemsets

# File path to your CSV dataset
file_path = r"E:\4-2 term\DM Lab\Data-Mining-Lab\Lab 2\archive\groceries_dataset.csv"
dataset = load_dataset(file_path)

# Minimum support
min_support = 4

# Run the Apriori algorithm
frequent_itemsets = apriori_algorithm(dataset, min_support)

# Print all frequent itemsets
#print("\nAll frequent itemsets with min_support = {}:".format(min_support))
for idx, (itemset, support) in enumerate(frequent_itemsets.items(), start=1):
    itemset_str = ', '.join(itemset)
    #print(f"Item {idx}: {itemset_str}, Support: {support}")
