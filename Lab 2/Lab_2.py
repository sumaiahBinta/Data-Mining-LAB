from itertools import combinations

# Function to load the dataset from a CSV file
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            transaction = line.strip().split(',')  # Assuming items in transactions are comma-separated
            dataset.append([item.strip() for item in transaction if item])
    return dataset

# Generate itemsets of size k
def generate_itemsets(transactions, k):
    items = set(item for transaction in transactions for item in transaction)
    return [set(comb) for comb in combinations(items, k)]

# Count the support for each itemset
def count_support(transactions, itemsets):
    support = {}
    for itemset in itemsets:
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        support[frozenset(itemset)] = count
    return support

# Prune itemsets based on minimum support
def prune_itemsets(itemset_support, min_support):
    return {itemset: support for itemset, support in itemset_support.items() if support >= min_support}

# Implement the Apriori algorithm
def apriori(transactions, min_support):
    k = 1
    frequent_itemsets = []
    while True:
        itemsets = generate_itemsets(transactions, k)
        if not itemsets:
            break
        itemset_support = count_support(transactions, itemsets)
        frequent_itemset = prune_itemsets(itemset_support, min_support)
        if not frequent_itemset:
            break
        frequent_itemsets.append(frequent_itemset)
        k += 1
    return frequent_itemsets

# Find the closed itemsets
def find_closed_itemsets(frequent_itemsets, itemset_support):
    closed_itemsets = []
    for itemset in frequent_itemsets:
        is_closed = True
        for other_itemset in frequent_itemsets:
            if itemset < other_itemset and itemset_support[itemset] == itemset_support[other_itemset]:
                is_closed = False
                break
        if is_closed:
            closed_itemsets.append(itemset)
    return closed_itemsets

# Find the maximal itemsets
def find_maximal_itemsets(frequent_itemsets):
    maximal_itemsets = []
    for itemset in frequent_itemsets:
        is_maximal = True
        for other_itemset in frequent_itemsets:
            if itemset < other_itemset:
                is_maximal = False
                break
        if is_maximal:
            maximal_itemsets.append(itemset)
    return maximal_itemsets

# Optimal min_support finding
def find_optimal_min_support(dataset):
    optimal_support = None
    optimal_patterns = None
    min_difference = float('inf')

    for min_support in range(1, 6):  # Adjust the range if needed
        frequent_itemsets = apriori(dataset, min_support)
        total_patterns = sum(len(k_itemset) for k_itemset in frequent_itemsets)
        print(f"\nMin Support: {min_support}, Number of frequent patterns: {total_patterns}")

        if 5 <= total_patterns <= 20:  # Define acceptable range for "optimal"
            current_diff = abs(10 - total_patterns)  # Aim for 10 frequent patterns
            if current_diff < min_difference:
                min_difference = current_diff
                optimal_support = min_support
                optimal_patterns = frequent_itemsets

    return optimal_support, optimal_patterns

# Main function
def main():
    file_path = "archive/groceries.csv"  # Dataset path

    # Step 1: Load dataset from CSV file
    dataset = load_dataset(file_path)

    # Step 2: Apply Apriori algorithm
    min_support = 2
    frequent_itemsets = apriori(dataset, min_support)

    # Step 3: Display frequent itemsets
    for idx, k_itemset in enumerate(frequent_itemsets, start=1):
        print(f"Frequent {idx}-itemsets:")
        for itemset, support in k_itemset.items():
            print(f"Itemset: {set(itemset)}, Support: {support}")

    # Step 4: Find closed and maximal itemsets
    all_itemsets = []
    for k_itemset in frequent_itemsets:
        all_itemsets.extend(list(k_itemset.keys()))

    itemset_support = count_support(dataset, all_itemsets)
    closed_itemsets = find_closed_itemsets(all_itemsets, itemset_support)
    maximal_itemsets = find_maximal_itemsets(all_itemsets)

    print("\nClosed Itemsets:")
    for itemset in closed_itemsets:
        print(set(itemset))

    print("\nMaximal Itemsets:")
    for itemset in maximal_itemsets:
        print(set(itemset))

    # Step 5: Find the optimal min_support
    optimal_support, optimal_patterns = find_optimal_min_support(dataset)
    print(f"\nOptimal Min Support: {optimal_support}")
    for idx, k_itemset in enumerate(optimal_patterns, start=1):
        print(f"Frequent {idx}-itemsets:")
        for itemset, support in k_itemset.items():
            print(f"Itemset: {set(itemset)}, Support: {support}")

if __name__ == "__main__":
    main()
