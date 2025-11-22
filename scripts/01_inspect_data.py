import sys
import os

# Add parent directory to path to allow importing from config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
import logging
from config.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def inspect_data():
    logger.info("Loading dataset '0x404/ccs_dataset'...")
    try:
        # Load the dataset
        dataset = load_dataset("0x404/ccs_dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    logger.info("\nDataset loaded successfully!")
    logger.info(f"Structure: {dataset}")

    # Inspect the first split (usually 'train')
    split_name = list(dataset.keys())[0]
    logger.info(f"\nInspecting split: '{split_name}'")
    logger.info(f"Number of examples: {len(dataset[split_name])}")
    logger.info(f"Features: {dataset[split_name].features}")

    # Get label info (using 'type' column)
    if 'type' in dataset[split_name].features:
        logger.info("\nFound 'type' column. Analyzing unique values...")
        unique_types = set(dataset[split_name]['type'])
        logger.info(f"Unique Labels found ({len(unique_types)}): {unique_types}")
        
        # Create a simple mapping for display
        logger.info("\nProposed Label Mappings:")
        sorted_types = sorted(list(unique_types))
        for idx, name in enumerate(sorted_types):
            logger.info(f"  ID {idx} -> {name}")
    else:
        logger.info("\nDataset does not contain a 'type' column.")

    # Show a few examples
    logger.info("\nFirst 3 Examples:")
    for i in range(3):
        example = dataset[split_name][i]
        label = example['type'] if 'type' in example else 'UNKNOWN'
        text = example['commit_message'] if 'commit_message' in example else 'UNKNOWN'
        logger.info(f"  [{i}] Label: {label} | Text: {text}")

if __name__ == "__main__":
    inspect_data()
