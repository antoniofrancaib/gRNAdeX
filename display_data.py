import os
import torch

# Set the path to your data directory
DATA_PATH = "data/" 

def load_and_print_data():
    # Load data from the processed.pt file
    data_list = list(torch.load(os.path.join(DATA_PATH, "processed_filtered.pt")).values())

    # Print the first few entries to see how they look
    print(f"Sample {3}: {len(data_list)}")

if __name__ == "__main__":
    load_and_print_data()