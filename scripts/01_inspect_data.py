import sys
from datasets import load_dataset

def inspect_data():
    print("Loading dataset '0x404/ccs_dataset'...")
    try:
        # Load the dataset
        dataset = load_dataset("0x404/ccs_dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print("\nDataset loaded successfully!")
    print(f"Structure: {dataset}")

    # Inspect the first split (usually 'train')
    split_name = list(dataset.keys())[0]
    print(f"\nInspecting split: '{split_name}'")
    print(f"Number of examples: {len(dataset[split_name])}")
    print(f"Features: {dataset[split_name].features}")

    # Get label info (using 'type' column)
    if 'type' in dataset[split_name].features:
        print("\nFound 'type' column. Analyzing unique values...")
        unique_types = set(dataset[split_name]['type'])
        print(f"Unique Labels found ({len(unique_types)}): {unique_types}")
        
        # Create a simple mapping for display
        print("\nProposed Label Mappings:")
        sorted_types = sorted(list(unique_types))
        for idx, name in enumerate(sorted_types):
            print(f"  ID {idx} -> {name}")
    else:
        print("\nDataset does not contain a 'type' column.")

    # Show a few examples
    print("\nFirst 3 Examples:")
    for i in range(3):
        example = dataset[split_name][i]
        label = example['type'] if 'type' in example else 'UNKNOWN'
        text = example['commit_message'] if 'commit_message' in example else 'UNKNOWN'
        print(f"  [{i}] Label: {label} | Text: {text}")

if __name__ == "__main__":
    inspect_data()
